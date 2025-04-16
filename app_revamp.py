from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from connect4 import Connect4
from mcts import ucb2_agent

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
        self.game = Connect4()
        self.strategy = ucb2_agent(7)
        self.pos = self.game.get_initial_position()
        self.board = [[0 for _ in range(7)] for _ in range(6)]

    def convert_board_format(self, json_board):
        """Convert JSON board to the format expected by Connect4"""
        # The JSON board is already in the right format (2D array)
        # Just making a copy to be safe
        return [row[:] for row in json_board]
    
    def board_move(self, col, turn):
        for i in range(5, -1, -1):
            if self.board[i][col] == 0:
                self.board[i][col] = 1 if turn == 0 else 2
                return i
        return None

    def handle_move(self, move):
        row = self.board_move(move, self.pos.turn)
        if row is not None:
            self.pos = self.pos.move(move)
            self.draw_grid()
            if self.check_game_over():
                return

    def get_move(self, game_state: GameState):
        self.board = game_state.board
            
  # Create a global agent instance
connect4_agent = Connect4Agent()

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        # Log the received game state
        print(f"Received game state: {game_state.dict()}")
        
        # Validate the input
        if not game_state.valid_moves:
            raise HTTPException(status_code=400, detail="No valid moves available")
        
        # Get the best move using our agent
        selected_move = connect4_agent.get_move(game_state)
        
        # Log the selected move
        print(f"Selected move: {selected_move}")
        
        # Return the selected move
        return AIResponse(move=selected_move)
        
    except Exception as e:
        # In case of any error, try to return a valid move
        if game_state.valid_moves:
            fallback_move = game_state.valid_moves[0]
            print(f"Exception: {str(e)}. Using fallback move: {fallback_move}")
            return AIResponse(move=fallback_move)
        
        # If no valid moves, raise an HTTP exception
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)