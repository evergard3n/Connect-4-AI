# import random
# import time
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Tuple, Dict, Optional
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

# # Game constants
# ROWS = 6
# COLS = 7
# WIN_LENGTH = 4
# MAX_DEPTH = 8
# TIME_LIMIT = 3.0
# ASP_WIN = 50             # aspiration window size
# FOUR_IN_ROW = 100_000_000  # win score threshold
# THREE_IN_ROW = 1000
# TWO_IN_ROW = 100

# # Zobrist hashing setup
# ZOBRIST = [[[random.getrandbits(64) for _ in range(2)] for _ in range(COLS)] for _ in range(ROWS)]
# EMPTY_HASH = random.getrandbits(64)

# # FastAPI setup
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class GameState(BaseModel):
#     board: List[List[int]]
#     current_player: int
#     valid_moves: List[int]

# class AIResponse(BaseModel):
#     move: int
#     evaluation: Optional[int] = None
#     depth: Optional[int] = None
#     execution_time: Optional[float] = None

# class BitboardState:
#     def __init__(self, board: List[List[int]]):
#         # Store 2D board for evaluation
#         self.board = [row.copy() for row in board]
#         self.bb = [0, 0]
#         self.zhash = EMPTY_HASH
#         self.heights = [-1] * COLS
#         # build bitboards and compute heights
#         for c in range(COLS):
#             h = -1
#             for r in range(ROWS-1, -1, -1):
#                 p = self.board[r][c]
#                 if p == 0 and h == -1:
#                     h = r
#                 if p in (1,2):
#                     pos = r * COLS + c
#                     self.bb[p-1] |= (1 << pos)
#                     self.zhash ^= ZOBRIST[r][c][p-1]
#             self.heights[c] = h

#     def make_move(self, col: int, player: int) -> int:
#         row = self.heights[col]
#         if row < 0:
#             return -1
#         # update structures
#         self.board[row][col] = player
#         pos = row * COLS + col
#         self.bb[player-1] |= (1 << pos)
#         self.zhash ^= ZOBRIST[row][col][player-1]
#         # update height
#         nh = -1
#         for r in range(row-1, -1, -1):
#             if self.board[r][col] == 0:
#                 nh = r
#                 break
#         self.heights[col] = nh
#         return row

#     def undo_move(self, col: int, player: int, row: int):
#         # revert structures
#         self.board[row][col] = 0
#         pos = row * COLS + col
#         self.bb[player-1] ^= (1 << pos)
#         self.zhash ^= ZOBRIST[row][col][player-1]
#         self.heights[col] = row

#     def valid_moves(self) -> List[int]:
#         return [c for c in range(COLS) if self.heights[c] >= 0]

#     def is_win(self, player: int) -> bool:
#         b = self.bb[player-1]
#         # horizontal
#         m = b & (b >> 1)
#         if m & (m >> 2): return True
#         # vertical
#         m = b & (b >> COLS)
#         if m & (m >> (2*COLS)): return True
#         # diag / (COLS+1)
#         m = b & (b >> (COLS+1))
#         if m & (m >> (2*(COLS+1))): return True
#         # diag \ (COLS-1)
#         m = b & (b >> (COLS-1))
#         if m & (m >> (2*(COLS-1))): return True
#         return False

#     def evaluate_position(self, player: int) -> int:
#         """Evaluate board using original heuristic"""
#         score = 0
#         # center column control
#         center = COLS // 2
#         center_count = sum(1 for r in range(ROWS) if self.board[r][center] == player)
#         score += center_count * 3
#         # count windows
#         for r in range(ROWS):
#             for c in range(COLS - WIN_LENGTH + 1):
#                 window = [self.board[r][c+i] for i in range(WIN_LENGTH)]
#                 score += self.evaluate_window(window, player)
#         for c in range(COLS):
#             for r in range(ROWS - WIN_LENGTH + 1):
#                 window = [self.board[r+i][c] for i in range(WIN_LENGTH)]
#                 score += self.evaluate_window(window, player)
#         for r in range(ROWS - WIN_LENGTH + 1):
#             for c in range(COLS - WIN_LENGTH + 1):
#                 window = [self.board[r+i][c+i] for i in range(WIN_LENGTH)]
#                 score += self.evaluate_window(window, player)
#         for r in range(WIN_LENGTH - 1, ROWS):
#             for c in range(COLS - WIN_LENGTH + 1):
#                 window = [self.board[r-i][c+i] for i in range(WIN_LENGTH)]
#                 score += self.evaluate_window(window, player)
#         return score

#     @staticmethod
#     def evaluate_window(window: List[int], player: int) -> int:
#         opponent = 3 - player
#         pc = window.count(player)
#         oc = window.count(opponent)
#         ec = window.count(0)
#         if pc > 0 and oc > 0:
#             return 0
#         if pc == 4:
#             return FOUR_IN_ROW
#         if pc == 3 and ec == 1:
#             return THREE_IN_ROW
#         if pc == 2 and ec == 2:
#             return TWO_IN_ROW
#         if oc == 3 and ec == 1:
#             return -THREE_IN_ROW
#         if oc == 2 and ec == 2:
#             return -TWO_IN_ROW
#         return 0

# class Connect4AI:
#     def __init__(self):
#         self.killer = [[None]*2 for _ in range(MAX_DEPTH+1)]
#         self.history = [[0]*COLS for _ in range(MAX_DEPTH+1)]
#         self.tt: Dict[int, Tuple[int,int,int,bool]] = {}

#     def detect_threats(self, state: BitboardState, player: int) -> List[int]:
#         threats = []
#         for col in state.valid_moves():
#             row = state.make_move(col, player)
#             if row >= 0 and state.is_win(player):
#                 threats.append(col)
#             if row >= 0:
#                 state.undo_move(col, player, row)
#         return threats

#     def search(self, state: BitboardState, depth: int, alpha: int, beta: int,
#                player: int, start: float, tlim: float) -> Tuple[Optional[int], Optional[int]]:
#         if time.time() - start > tlim:
#             return None, None
#         key = state.zhash
#         if key in self.tt:
#             sc, d, mv, exact = self.tt[key]
#             if d >= depth and exact:
#                 return sc, mv
#         if state.is_win(player):
#             return FOUR_IN_ROW, None
#         if state.is_win(3-player):
#             return -FOUR_IN_ROW, None
#         if depth == 0:
#             return state.evaluate_position(player), None
#         # move ordering
#         moves = state.valid_moves()
#         km = self.killer[depth][0]
#         if km in moves:
#             moves.remove(km)
#             moves.insert(0, km)
#         moves.sort(key=lambda c: self.history[depth][c], reverse=True)

#         best_val, best_mv = -10**9, moves[0] if moves else None
#         for m in moves:
#             row = state.make_move(m, player)
#             sc, _ = self.search(state, depth-1, -beta, -alpha, 3-player, start, tlim)
#             state.undo_move(m, player, row)
#             if sc is None:
#                 return None, None
#             val = -sc
#             if val > best_val:
#                 best_val, best_mv = val, m
#             alpha = max(alpha, val)
#             if alpha >= beta:
#                 self.killer[depth][1] = self.killer[depth][0]
#                 self.killer[depth][0] = m
#                 self.history[depth][m] += 1 << depth
#                 break
#         self.tt[key] = (best_val, depth, best_mv, True)
#         return best_val, best_mv

#     def find_best(self, board: List[List[int]], player: int, valid_moves: List[int]) -> Tuple[int,int,int,float]:
#         state = BitboardState(board)
#         moves = state.valid_moves()
#         start = time.time()
#         # 1) win in one
#         win_moves = self.detect_threats(state, player)
#         if win_moves:
#             win_moves.sort(key=lambda c: abs(c - COLS//2))
#             return win_moves[0], FOUR_IN_ROW, 1, time.time() - start
#         # 2) block opponent
#         opp_moves = self.detect_threats(state, 3-player)
#         if len(opp_moves) == 1:
#             return opp_moves[0], -FOUR_IN_ROW, 1, time.time() - start
#         # iterative deepening
#         best, best_sc, best_d = moves[0], 0, 0
#         prev_sc = 0
#         self.tt.clear()
#         for d in range(1, MAX_DEPTH+1):
#             if time.time() - start > TIME_LIMIT * 0.9:
#                 break
#             a, b = (-10**9, 10**9) if d == 1 else (prev_sc - ASP_WIN, prev_sc + ASP_WIN)
#             sc, mv = self.search(state, d, a, b, player, start, TIME_LIMIT * 0.9)
#             if sc is None:
#                 break
#             if sc <= a or sc >= b:
#                 sc, mv = self.search(state, d, -10**9, 10**9, player, start, TIME_LIMIT * 0.9)
#             if mv in moves:
#                 best, best_sc, best_d, prev_sc = mv, sc, d, sc
#             if best_sc >= FOUR_IN_ROW // 2:
#                 break
#         return best, best_sc, best_d, time.time() - start

# ai = Connect4AI()

# @app.post("/api/connect4-move")
# async def move(game: GameState) -> AIResponse:
#     try:
#         m, sc, dp, tm = ai.find_best(game.board, game.current_player, game.valid_moves)
#         return AIResponse(move=m, evaluation=sc, depth=dp, execution_time=tm)
#     except Exception:
#         fallback = game.valid_moves[0] if game.valid_moves else 0
#         return AIResponse(move=fallback)

# @app.get("/api/test")
# async def health():
#     return {"status": "ok"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)

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
            uct_c=0.5,  # Exploration constant (higher values = more exploration)
            max_simulations=10000,  # Increase for stronger play, decrease for speed
            max_memory_mb=400,
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
            print(f"Opponent moved: {opp_col}, {opp_row}")
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
        print(next_move)
        return AIResponse(move=next_move)
    except Exception as e:
        # Fallback to the first valid move if something goes wrong
        if game_state.valid_moves:
            print(f"Error occurred, falling back to first valid move: {str(e)}")
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/test")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)