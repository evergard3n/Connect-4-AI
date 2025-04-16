from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from connect4 import Connect4
from mcts import ucb2_agent

app = FastAPI()
agent = ucb2_agent(7)
game = Connect4()

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

def board_to_bitmap(board):
    bitmap = 0
    for row in reversed(board):
        for cell in row:
            if cell != 0:
                bitmap |= 1 << (len(board[0]) * (len(board) - 1 - board.index(row)) + row.index(cell))
    return bitmap

class Connect4Agent:
    def __init__(self):
        self.game = Connect4()
        self.strategy = ucb2_agent(7)
        self.pos = self.game.get_initial_position()
        self.received_board = [[0 for _ in range(7)] for _ in range(6)]
        self.old_board = [[0 for _ in range(7)] for _ in range(6)]
    
    # def board_move(self, col, turn):
    #     for i in range(5, -1, -1):
    #         if self.board[i][col] == 0:
    #             self.board[i][col] = 2 if turn == 0 else 1
    #             return
    
    def find_lastest_move(self): 
        for col in range(7):
            for row in range(6):
                if self.received_board[row][col] != 1 and self.received_board[row][col] != self.old_board[row][col]:
                    return col


    def create_position_from_game_state(self, game_state: GameState):
        # self.board = game_state.board
        self.received_board = game_state.board

        new_col = self.find_lastest_move()
        print('new_col:', new_col)
        self.old_board = game_state.board
        print('old_board:', self.old_board)
        if (new_col != None):
            self.pos = self.game.get_initial_position()
            self.pos.turn = 0
            self.pos = self.pos.move(new_col)

        self.pos.turn = 1
        move = self.strategy(self.pos)
        print('move:', move)
        self.pos = self.pos.move(move)
        return move
        

connect4agent = Connect4Agent()

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        # In ra GameState nhận được
        print("GameState nhận được:", game_state.dict())
        print('huh')
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
            
        next_move = connect4agent.create_position_from_game_state(game_state)
        print('got here: ', next_move)
        return AIResponse(move=next_move)
    except Exception as e:
        if game_state.valid_moves:
            # print(game_state.valid_moves[0])
            print('got some error')
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)