from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pyspiel
import copy
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

class Connect4Agent:
    def __init__(self):
        self.game = pyspiel.load_game("connect_four")
        self.state = self.game.new_initial_state()
        self.old_board = [[0 for _ in range(7)] for _ in range(6)]
        self.computer_moves_made = 0
        
        # Configure MCTS bot with optimal parameters for Connect4
        self.bot = pyspiel.MCTSBot(
            game=self.game,
            evaluator=pyspiel.RandomRolloutEvaluator(n_rollouts=20, seed=42),
            uct_c=1.5,  # Exploration constant (higher values = more exploration)
            max_simulations=10000,  # Increase for stronger play, decrease for speed
            max_memory_mb=1000,
            solve=True,  # Try to solve game states when possible
            seed=42,
            verbose=False
        )

    def detect_game_state(self, board):
        """Detect if this is a new game or continuation."""
        # Check if board is completely empty (new game)
        is_empty_board = all(cell == 0 for row in board for cell in row)
        
        # Check if board has only one piece (opponent went first in a new game)
        piece_count = sum(1 for row in board for cell in row if cell != 0)
        is_first_move = piece_count == 1
        
        return is_empty_board, is_first_move

    def find_opponent_move(self, current_board):
        """Find what move the opponent made since our last move."""
        for col in range(7):
            # Count pieces in this column before and after
            old_count = sum(1 for row in range(6) if self.old_board[row][col] != 0)
            new_count = sum(1 for row in range(6) if current_board[row][col] != 0)
            
            if new_count > old_count:
                # Find the exact row where piece was added
                for row in range(5, -1, -1):
                    if current_board[row][col] != 0 and self.old_board[row][col] == 0:
                        return col, row
        
        return None, None

    def handle_first_move(self, board):
        """Handle the case where opponent made the first move."""
        for col in range(7):
            for row in range(5, -1, -1):
                if board[row][col] != 0:
                    # Update our internal state
                    self.state.apply_action(col)
                    # Update our board tracking
                    self.old_board[row][col] = board[row][col]
                    return col
        
        return None

    def get_ai_move(self, gs: GameState) -> int:
        """Get the best move using MCTS with incremental state tracking."""
        is_empty_board, is_first_move = self.detect_game_state(gs.board)
        
        # CASE 1: Brand new game with empty board
        if is_empty_board:
            # Reset everything
            self.state = self.game.new_initial_state()
            self.old_board = [[0 for _ in range(7)] for _ in range(6)]
            self.computer_moves_made = 0
            
            # Center column is a strong opening move
            best_move = 3
            
            # Apply our move to our internal state
            self.state.apply_action(best_move)
            
            # Update our board tracking
            for row in range(5, -1, -1):
                if self.old_board[row][best_move] == 0:
                    self.old_board[row][best_move] = 2  # AI uses player 2
                    break
            
            self.computer_moves_made += 1
            return best_move
            
        # CASE 2: New game where opponent went first
        elif is_first_move:
            # Reset everything
            self.state = self.game.new_initial_state()
            self.old_board = [[0 for _ in range(7)] for _ in range(6)]
            self.computer_moves_made = 0
            
            # Apply opponent's first move to our state
            opponent_col = self.handle_first_move(gs.board)
            
            # Use MCTS for our response
            action = self.bot.step(self.state)
            
            # Apply our move to internal state
            self.state.apply_action(action)
            
            # Update our board tracking
            for row in range(5, -1, -1):
                if self.old_board[row][action] == 0:
                    self.old_board[row][action] = 2  # AI uses player 2
                    break
            
            self.computer_moves_made += 1
            return action
            
        # CASE 3: Continuing game - find opponent's last move and respond
        else:
            # Find opponent's move since last time
            opp_col, opp_row = self.find_opponent_move(gs.board)
            
            if opp_col is not None:
                # Update our tracking board
                self.old_board[opp_row][opp_col] = gs.board[opp_row][opp_col]
                
                # Apply opponent's move to our state
                if not self.state.is_terminal():
                    self.state.apply_action(opp_col)
            
            # Use MCTS to choose our move
            if not self.state.is_terminal():
                action = self.bot.step(self.state)
                
                # Apply our move to internal state
                self.state.apply_action(action)
                
                # Verify move is valid
                if action not in gs.valid_moves:
                    # Fallback if MCTS gives invalid move
                    action = gs.valid_moves[0]
                
                # Update our board tracking
                for row in range(5, -1, -1):
                    if self.old_board[row][action] == 0:
                        self.old_board[row][action] = 2  # AI uses player 2
                        break
                
                self.computer_moves_made += 1
                return action
            else:
                # Game is terminal, return a valid move if available
                if gs.valid_moves:
                    return gs.valid_moves[0]
                raise ValueError("Game is already completed")

# Initialize the agent
connect4_agent = Connect4Agent()

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        # Verify we have valid moves
        if not game_state.valid_moves:
            raise ValueError("No valid moves available")
            
        # Get the AI's move
        next_move = connect4_agent.get_ai_move(game_state)
        
        return AIResponse(move=next_move)
    except Exception as e:
        # Fallback to the first valid move if something goes wrong
        if game_state.valid_moves:
            print(f"Error occurred, falling back to first valid move: {str(e)}")
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)