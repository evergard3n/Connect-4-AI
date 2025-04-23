import fastapi
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pyspiel # Import open_spiel
import random # For fallback move
import traceback # For detailed error logging

# Ensure open_spiel is correctly installed and accessible

app = fastapi.FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models ---
class GameState(BaseModel):
    """Represents the game state received from the client."""
    board: List[List[int]] # 6 rows, 7 columns. 0: empty, 1: Player 1, 2: Player 2 (AI)
    current_player: int # Player whose turn it is (according to the client)
    valid_moves: List[int] # Columns where a piece can be dropped

class AIResponse(BaseModel):
    """Response containing the AI's chosen move."""
    move: int # Column index (0-6)

# --- Connect4 Agent using OpenSpiel MCTS ---
class Connect4Agent:
    """Manages the Connect Four game state and uses OpenSpiel MCTS for AI moves."""
    def __init__(self, simulations_per_move: int = 10000, uct_c: float = 0.5, rollout_count: int = 20, seed: int = 42):
        """
        Initializes the agent with specified MCTS parameters.

        Args:
            simulations_per_move: Number of MCTS simulations per move.
            uct_c: Exploration constant for the UCT algorithm in MCTS. Lower values favor exploitation.
            rollout_count: Number of rollouts per evaluation in the RandomRolloutEvaluator.
            seed: Random seed for reproducibility.
        """
        self.game = pyspiel.load_game("connect_four")
        # Define player IDs: API Player 1 -> OS Player 0, API Player 2 (AI) -> OS Player 1
        self.human_player_api = 1
        self.ai_player_api = 2
        self.human_player_os = 0
        self.ai_player_os = 1 # AI uses OpenSpiel player ID 1

        # Store config for potential reset
        self._simulations_per_move = simulations_per_move
        self._uct_c = uct_c
        self._rollout_count = rollout_count
        self._seed = seed

        print(f"Initializing MCTS Bot (Sims: {simulations_per_move}, UCT_C: {uct_c}, Rollouts: {rollout_count}, Seed: {seed})")

        # Configure the MCTS bot based on provided parameters and user example
        evaluator = pyspiel.RandomRolloutEvaluator(20,42)
        self.mcts_bot = pyspiel.MCTSBot(
            game=self.game,
            uct_c=self._uct_c,                      # Use provided uct_c
            max_simulations=self._simulations_per_move, # Use provided simulations
            evaluator=evaluator,                 # Use configured evaluator
            max_memory_mb=1000,                  # Added memory limit from example
            seed=self._seed,             # Use provided seed for the bot
            solve=True,                          # Use solve during search (from example)
            verbose=False                        # Keep verbose off unless debugging
            # child_selection_fn defaults to uct_value, which is standard
        )
        # Initialize the internal OpenSpiel game state
        self.state = self.game.new_initial_state()
        print("Connect4 Agent Initialized. Initial State:\n", self.state)

    def find_new_move(self, GameState):
        

# --- FastAPI Endpoint ---
# Initialize agent with desired parameters (matching user example)
connect4agent = Connect4Agent(
    simulations_per_move=10000,
    uct_c=0.5,
    rollout_count=20,
    seed=42
)

@app.post("/api/connect4-move", response_model=AIResponse)
async def make_move(game_state: GameState) -> AIResponse:
    """API endpoint to receive game state and return AI's move."""
    print(f"\n=== Received request: /api/connect4-move ===")
    print(f"Incoming Board:\n{np.array(game_state.board)}")
    print(f"Incoming Current Player (API): {game_state.current_player}")
    print(f"Incoming Valid Moves (API): {game_state.valid_moves}")

    # Basic validation
    if not game_state.board or len(game_state.board) != 6 or any(len(row) != 7 for row in game_state.board):
         print("Error: Invalid board dimensions.")
         raise fastapi.HTTPException(status_code=400, detail="Invalid board dimensions.")

    # Check for valid moves: If the client sends an empty list, it *might* mean the game ended *before* this request.
    # We rely on the agent's internal state check after synchronization for the definitive game over status.
    if not game_state.valid_moves:
         print("Warning: Client provided an empty list for valid_moves.")
         # The agent's get_ai_move will check is_terminal() after sync.

    try:
        # Get the AI's move using the agent
        ai_move = connect4agent.get_ai_move(game_state)

        # Basic validation of the returned move (should be 0-6)
        if not (0 <= ai_move <= 6):
             print(f"Error: Agent returned invalid move index: {ai_move}")
             # This indicates a serious internal error in the bot or state handling
             raise fastapi.HTTPException(status_code=500, detail=f"Internal agent error: Invalid move {ai_move} generated.")

        print(f"--- Sending Response ---")
        print(f"AI Move: {ai_move}")
        return AIResponse(move=ai_move)

    except ValueError as e:
        # Handle errors specifically raised by the agent (desync, game over, wrong turn, etc.)
        error_message = str(e)
        print(f"ValueError processing move: {error_message}")
        # Check for specific error messages if needed for different status codes
        if "Game has already ended" in error_message:
             raise fastapi.HTTPException(status_code=400, detail=error_message) # Bad request - game over
        elif "State desynchronized" in error_message or "Agent reset" in error_message:
             raise fastapi.HTTPException(status_code=409, detail=error_message) # Conflict - state issue
        elif "Not AI's turn" in error_message:
             raise fastapi.HTTPException(status_code=409, detail=error_message) # Conflict - turn issue
        else:
             raise fastapi.HTTPException(status_code=400, detail=f"Invalid request or game state: {error_message}")

    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"!!! Unexpected Internal Server Error !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc() # Log the full traceback for debugging

        # Reset agent state after a critical unknown error
        print("Resetting agent state due to critical unexpected error.")
        try:
            connect4agent.reset_state_and_bot()
        except Exception as reset_err:
            print(f"Error during agent reset after critical error: {reset_err}")

        # Return a generic 500 error
        raise fastapi.HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}. Agent has been reset.")


# --- Run the application ---
if __name__ == "__main__":
    print("Starting FastAPI server for Connect4 AI...")
    uvicorn.run(app, host="0.0.0.0", port=8080)

