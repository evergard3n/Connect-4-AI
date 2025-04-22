# # from fastapi import FastAPI, HTTPException
# # import random
# # import uvicorn
# # from pydantic import BaseModel
# # from typing import List, Optional
# # from fastapi.middleware.cors import CORSMiddleware
# # from connect4 import Connect4
# # from mcts import ucb2_agent

# # app = FastAPI()
# # agent = ucb2_agent(3)
# # game = Connect4()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # class GameState(BaseModel):
# #     board: List[List[int]]
# #     current_player: int
# #     valid_moves: List[int]

# # class AIResponse(BaseModel):
# #     move: int

# # def board_to_bitmap(board):
# #     bitmap = 0
# #     for row in reversed(board):
# #         for cell in row:
# #             if cell != 0:
# #                 bitmap |= 1 << (len(board[0]) * (len(board) - 1 - board.index(row)) + row.index(cell))
# #     return bitmap

# # class Connect4Agent:
# #     def __init__(self):
# #         self.game = Connect4()
# #         self.strategy = ucb2_agent(7)
# #         self.pos = self.game.get_initial_position()
# #         self.received_board = [[0 for _ in range(7)] for _ in range(6)]
# #         self.old_board = [[0 for _ in range(7)] for _ in range(6)]
    
# #     def board_move(self, col, turn):
# #         for i in range(5, -1, -1):
# #             if self.board[i][col] == 0:
# #                 self.board[i][col] = 2 if turn == 0 else 1
# #                 return i
# #         return None
    
# #     def find_lastest_move(self): 
# #         for col in range(7):
# #             for row in range(6):
# #                 if self.received_board[row][col] != 1 and self.received_board[row][col] != self.old_board[row][col]:
# #                     return col


# #     def create_position_from_game_state(self, game_state: GameState):
# #         # self.board = game_state.board
# #         self.received_board = game_state.board

# #         new_col = self.find_lastest_move()
# #         print('new_col:', new_col)
# #         self.old_board = game_state.board
# #         print('old_board:', self.old_board)
# #         if (new_col != None and 0 <= new_col < 7 and self.old_board[0][new_col] == 0):
# #             # self.pos = self.game.get_initial_position()
# #             self.pos.turn = 0
# #             self.pos = self.pos.move(new_col)
# #         else:
# #             self.pos = self.game.get_initial_position()
# #         self.pos.turn = 1
# #         move = self.strategy(self.pos)
# #         print('move:', move)
# #         self.pos = self.pos.move(move)
# #         return move
        

# # connect4agent = Connect4Agent()

# # @app.post("/api/connect4-move")
# # async def make_move(game_state: GameState) -> AIResponse:
# #     try:
# #         # In ra GameState nhận được
# #         print("GameState nhận được:", game_state.dict())
# #         print('huh')
# #         if not game_state.valid_moves:
# #             raise ValueError("Không có nước đi hợp lệ")
            
# #         next_move = connect4agent.create_position_from_game_state(game_state)
# #         print('got here: ', next_move)
# #         return AIResponse(move=next_move)
# #     except Exception as e:
# #         if game_state.valid_moves:
# #             # print(game_state.valid_moves[0])
# #             print('got some error')
# #             return AIResponse(move=game_state.valid_moves[0])
# #         raise HTTPException(status_code=400, detail=str(e))

# # @app.get("/api/test")
# # async def health_check():
# #     return {"status": "ok", "message": "Server is running"}

# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8080)
# from fastapi import FastAPI, HTTPException
# import uvicorn
# from pydantic import BaseModel
# from typing import List, Optional, Tuple, Dict
# from fastapi.middleware.cors import CORSMiddleware
# import math
# import time
# import random
# from functools import lru_cache

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Game constants
# ROWS = 6
# COLS = 7
# WIN_LENGTH = 4
# MAX_DEPTH = 8  # Increased for better lookahead
# TIME_LIMIT = 3  # Seconds to ensure we respond within 5s limit

# # Define key pattern values for faster evaluation
# FOUR_IN_ROW = 100000000
# THREE_IN_ROW = 1000
# TWO_IN_ROW = 100
# BLOCK_THREE = 1200
# BLOCK_TWO = 100

# class GameState(BaseModel):
#     board: List[List[int]]
#     current_player: int
#     valid_moves: List[int]

# class AIResponse(BaseModel):
#     move: int
#     evaluation: Optional[int] = None
#     depth: Optional[int] = None
#     execution_time: Optional[float] = None

# class Connect4AI:
#     # Cache for lines and transposition table
#     _cached_winning_lines: Optional[List[List[Tuple[int, int]]]] = None
#     _transposition_table: Dict[str, Tuple[int, int, int, bool]] = {}  # hash -> (score, depth, move, is_exact)
    
#     @staticmethod
#     def board_hash(board: List[List[int]]) -> str:
#         """Create a unique string representation of the board for caching"""
#         return ''.join(''.join(str(cell) for cell in row) for row in board)
    
#     @staticmethod
#     def get_winning_lines() -> List[List[Tuple[int, int]]]:
#         """Get all potential winning lines (cached)"""
#         if Connect4AI._cached_winning_lines is not None:
#             return Connect4AI._cached_winning_lines

#         lines = []
#         # Horizontal
#         for r in range(ROWS):
#             for c in range(COLS - WIN_LENGTH + 1):
#                 lines.append([(r, c + i) for i in range(WIN_LENGTH)])
#         # Vertical
#         for c in range(COLS):
#             for r in range(ROWS - WIN_LENGTH + 1):
#                 lines.append([(r + i, c) for i in range(WIN_LENGTH)])
#         # Diagonal down-right
#         for r in range(ROWS - WIN_LENGTH + 1):
#             for c in range(COLS - WIN_LENGTH + 1):
#                 lines.append([(r + i, c + i) for i in range(WIN_LENGTH)])
#         # Diagonal up-right
#         for r in range(WIN_LENGTH - 1, ROWS):
#             for c in range(COLS - WIN_LENGTH + 1):
#                 lines.append([(r - i, c + i) for i in range(WIN_LENGTH)])
#         Connect4AI._cached_winning_lines = lines
#         return lines

#     @staticmethod
#     def evaluate_window(window: List[int], player: int) -> int:
#         """Evaluate a window of 4 positions"""
#         opponent = 3 - player
#         player_count = window.count(player)
#         opponent_count = window.count(opponent)
#         empty_count = window.count(0)
        
#         # No score if both players have pieces in the window
#         if player_count > 0 and opponent_count > 0:
#             return 0
            
#         # Player pieces evaluation
#         if player_count == 4:
#             return FOUR_IN_ROW
#         elif player_count == 3 and empty_count == 1:
#             return THREE_IN_ROW
#         elif player_count == 2 and empty_count == 2:
#             return TWO_IN_ROW
#         elif player_count == 1 and empty_count == 3:
#             return 1
            
#         # Opponent pieces evaluation (negative scores)
#         if opponent_count == 3 and empty_count == 1:
#             return -BLOCK_THREE  # Critical to block
#         elif opponent_count == 2 and empty_count == 2:
#             return -BLOCK_TWO
#         elif opponent_count == 1 and empty_count == 3:
#             return -1
            
#         return 0

#     @staticmethod
#     def evaluate_position(board: List[List[int]], player: int) -> int:
#         """Comprehensive board evaluation function"""
#         score = 0
#         center_col = COLS // 2
        
#         # Center column control is strategically important
#         center_array = [board[r][center_col] for r in range(ROWS)]
#         center_player_count = center_array.count(player)
#         score += center_player_count * 3
        
#         # Evaluate all potential winning lines
#         winning_lines = Connect4AI.get_winning_lines()
#         for line in winning_lines:
#             window = [board[r][c] for r, c in line]
#             score += Connect4AI.evaluate_window(window, player)
        
#         # Evaluate positional advantages
#         for c in range(COLS):
#             for r in range(ROWS):
#                 if board[r][c] == player:
#                     # Prefer positions closer to center horizontally
#                     distance_from_center = abs(c - center_col)
#                     score += (3 - min(3, distance_from_center)) * 5
                    
#                     # Prefer lower positions (more stable and enable more builds)
#                     score += (ROWS - r) * 3
                    
#                     # Check for connected pieces (stronger positions)
#                     # Horizontal connections
#                     if c > 0 and board[r][c-1] == player:
#                         score += 3
#                     if c < COLS-1 and board[r][c+1] == player:
#                         score += 3
                    
#                     # Vertical connections
#                     if r < ROWS-1 and board[r+1][c] == player:
#                         score += 5  # Vertical builds are strong
                        
#                     # Diagonal connections
#                     if r > 0 and c > 0 and board[r-1][c-1] == player:
#                         score += 2
#                     if r > 0 and c < COLS-1 and board[r-1][c+1] == player:
#                         score += 2
#                     if r < ROWS-1 and c > 0 and board[r+1][c-1] == player:
#                         score += 2
#                     if r < ROWS-1 and c < COLS-1 and board[r+1][c+1] == player:
#                         score += 2
        
#         return score

#     @staticmethod
#     def check_winner(board: List[List[int]]) -> int:
#         """Determine if there's a winner on the board"""
#         winning_lines = Connect4AI.get_winning_lines()
#         for line in winning_lines:
#             values = [board[r][c] for r, c in line]
#             if values[0] != 0 and values.count(values[0]) == 4:
#                 return values[0]
#         return 0

#     @staticmethod
#     def detect_threats(board: List[List[int]], player: int) -> List[int]:
#         """Detect columns that represent immediate threats (win next move)"""
#         threat_columns = []
    
#     # Iterate over all columns to check if they are valid for a move
#         for col in range(COLS):
#             # Skip if column is full
#             if board[0][col] != 0:
#                 continue
                
#             # Simulate the move for the current player in this column
#             new_board, row = Connect4AI.make_move(board, col, player)
            
#             # Check if the current move creates a winning position
#             if Connect4AI.check_winner(new_board) == player:
#                 threat_columns.append(col)
                
#         return threat_columns
#     @staticmethod
#     def is_board_full(board: List[List[int]]) -> bool:
#         """Check if board is full (draw)"""
#         return all(board[0][c] != 0 for c in range(COLS))

#     @staticmethod
#     def is_terminal_node(board: List[List[int]]) -> bool:
#         """Check if position is terminal (game over)"""
#         return Connect4AI.check_winner(board) != 0 or Connect4AI.is_board_full(board)

#     @staticmethod
#     def get_valid_moves(board: List[List[int]]) -> List[int]:
#         """Get valid moves (empty columns)"""
#         return [c for c in range(COLS) if board[0][c] == 0]

#     @staticmethod
#     def get_next_open_row(board: List[List[int]], col: int) -> int:
#         """Find next open row in a column"""
#         for r in range(ROWS-1, -1, -1):
#             if board[r][col] == 0:
#                 return r
#         return -1

#     @staticmethod
#     def make_move(board: List[List[int]], col: int, player: int) -> Tuple[List[List[int]], int]:
#         """Make a move and return new board state and row"""
#         row = Connect4AI.get_next_open_row(board, col)
#         if row == -1:
#             return board, -1
#         new_board = [row.copy() for row in board]
#         new_board[row][col] = player
#         return new_board, row

#     @staticmethod
#     def order_moves(board: List[List[int]], valid_moves: List[int], player: int) -> List[int]:
#         """Order moves for better alpha-beta pruning efficiency"""
#         move_scores = []
        
#         # First check for immediate winning moves
#         winning_moves = Connect4AI.detect_threats(board, player)
#         if winning_moves:
#             # Prioritize center winning moves
#             center_col = COLS // 2
#             winning_moves.sort(key=lambda c: abs(c - center_col))
#             # Add remaining moves after winning moves
#             remaining_moves = [c for c in valid_moves if c not in winning_moves]
#             return winning_moves + remaining_moves
        
#         # Then check for opponent winning moves to block
#         blocking_moves = Connect4AI.detect_threats(board, 3 - player)
        
#         # Score moves based on position and potential
#         for col in valid_moves:
#             score = 0
            
#             # Blocking moves get high priority
#             if col in blocking_moves:
#                 score += 1000
                
#             # Prefer center and nearby columns
#             center_preference = COLS // 2
#             score += (4 - min(3, abs(col - center_preference))) * 10
            
#             # Check if move leads to good position
#             new_board, row = Connect4AI.make_move(board, col, player)
#             if row != -1:
#                 # Small sample evaluation
#                 score += Connect4AI.evaluate_position(new_board, player) // 1000
                
#                 # Avoid moves that give opponent winning moves
#                 opponent_threats = Connect4AI.detect_threats(new_board, 3 - player)
#                 if opponent_threats:
#                     score -= 500
                    
#                 # Look for two-move win setups
#                 for next_col in range(COLS):
#                     if next_col != col and next_col < COLS:
#                         if board[0][next_col] != 0:
#                             continue
#                         next_board, _ = Connect4AI.make_move(new_board, next_col, player)
#                         if Connect4AI.check_winner(next_board) == player:
#                             score += 50
            
#             move_scores.append((score, col))
            
#         # Sort by score descending
#         move_scores.sort(reverse=True, key=lambda x: x[0])
#         return [col for _, col in move_scores]

#     @staticmethod
#     def negamax_alpha_beta(board: List[List[int]], depth: int, alpha: float, beta: float, 
#                           player: int, start_time: float, time_limit: float) -> Tuple[int, Optional[int]]:
#         """Negamax with alpha-beta pruning and time management"""
#         # Check if we're running out of time
#         if time.time() - start_time > time_limit:
#             return None, None  # Signal we need to stop search
            
#         # Check transposition table
#         board_key = Connect4AI.board_hash(board)
#         if board_key in Connect4AI._transposition_table:
#             score, stored_depth, move, is_exact = Connect4AI._transposition_table[board_key]
#             if stored_depth >= depth and is_exact:
#                 return score, move
        
#         # Check for terminal states
#         winner = Connect4AI.check_winner(board)
#         if winner == player:
#             return FOUR_IN_ROW, None
#         elif winner == 3 - player:
#             return -FOUR_IN_ROW, None
#         elif Connect4AI.is_board_full(board):
#             return 0, None
        
#         # Depth limit reached
#         if depth == 0:
#             eval_score = Connect4AI.evaluate_position(board, player)
#             return eval_score, None
            
#         valid_moves = Connect4AI.get_valid_moves(board)
#         if not valid_moves:
#             return 0, None
            
#         # Order moves for better pruning
#         ordered_moves = Connect4AI.order_moves(board, valid_moves, player)
        
#         best_value = -math.inf
#         best_move = ordered_moves[0]  # Default to first move
        
#         # Try each move
#         for col in ordered_moves:
#             new_board, row = Connect4AI.make_move(board, col, player)
#             if row == -1:
#                 continue
                
#             # Opponent's turn (negative of opponent's best score)
#             value, _ = Connect4AI.negamax_alpha_beta(
#                 new_board, depth-1, -beta, -alpha, 3 - player, start_time, time_limit
#             )
            
#             # Check if search was aborted due to time
#             if value is None:
#                 return None, None
                
#             value = -value  # Negamax negation
            
#             if value > best_value:
#                 best_value = value
#                 best_move = col
                
#             alpha = max(alpha, best_value)
#             if alpha >= beta:
#                 break  # Beta cutoff
        
#         # Store in transposition table
#         Connect4AI._transposition_table[board_key] = (best_value, depth, best_move, True)
#         return best_value, best_move

#     @staticmethod
#     def find_best_move(board: List[List[int]], player: int, valid_moves: List[int]) -> Tuple[int, int, int, float]:
#         """Find best move using iterative deepening with time control"""
#         if not valid_moves:
#             raise ValueError("No valid moves available")
            
#         start_time = time.time()
#         best_move = valid_moves[0]  # Default to first valid move
#         best_score = -math.inf
#         max_depth_reached = 0
        
#         # Clear transposition table for new search
#         Connect4AI._transposition_table.clear()
        
#         # First check immediate threats
        
#         # Win in one move if possible
#         winning_moves = Connect4AI.detect_threats(board, player)
#         if winning_moves:
#             # Choose center-most winning move
#             winning_moves.sort(key=lambda c: abs(c - COLS//2))
#             return winning_moves[0], FOUR_IN_ROW, 1, time.time() - start_time
            
#         # Block opponent win
#         opponent_threats = Connect4AI.detect_threats(board, 3 - player)
#         if opponent_threats:
#             # Choose center-most blocking move
#             opponent_threats.sort(key=lambda c: abs(c - COLS//2))
#             return opponent_threats[0], -BLOCK_THREE, 1, time.time() - start_time
        
#         # Use iterative deepening to find best move within time limit
#         for depth in range(1, MAX_DEPTH + 1):
#             try:
#                 score, move = Connect4AI.negamax_alpha_beta(
#                     board, depth, -math.inf, math.inf, player, 
#                     start_time, TIME_LIMIT * 0.9  # Use 90% of time limit
#                 )
                
#                 # Check if search was aborted due to time
#                 if score is None:
#                     break
                    
#                 # Update best move if valid
#                 if move in valid_moves:
#                     best_move = move
#                     best_score = score
#                     max_depth_reached = depth
                    
#                 # If we found a winning move, no need to search deeper
#                 if best_score >= FOUR_IN_ROW // 2:
#                     break
                    
#             except Exception as e:
#                 print(f"Error at depth {depth}: {str(e)}")
#                 break
                
#             # Break if we're getting close to time limit
#             if time.time() - start_time > TIME_LIMIT * 0.8:
#                 break
        
#         return best_move, best_score, max_depth_reached, time.time() - start_time

# @app.post("/api/connect4-move")
# async def make_move(game_state: GameState) -> AIResponse:
#     try:
#         start_time = time.time()
#         valid_moves = game_state.valid_moves
#         print(game_state.board)
#         print(game_state.current_player)
#         print(game_state.valid_moves)
#         if not valid_moves:
#             raise ValueError("No valid moves available")
            
#         player = game_state.current_player
        
#         # Find best move
#         best_move, score, depth, calc_time = Connect4AI.find_best_move(
#             game_state.board, player, valid_moves
#         )
        
#         # Failsafe: Check if returned move is valid
#         if best_move not in valid_moves:
#             # Try center column first
#             center = COLS // 2
#             if center in valid_moves:
#                 best_move = center
#             else:
#                 best_move = valid_moves[0]
        
#         return AIResponse(
#             move=best_move,
#             evaluation=score,
#             depth=depth,
#             execution_time=calc_time
#         )
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         # Fallback to center or first valid move
#         if game_state.valid_moves:
#             center = COLS // 2
#             if center in game_state.valid_moves:
#                 return AIResponse(move=center)
#             return AIResponse(move=game_state.valid_moves[0])
#         raise HTTPException(status_code=400, detail=str(e))


# @app.get("/api/test")
# async def health_check():
#     return {"status": "ok", "message": "Server is running"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)



from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
from fastapi.middleware.cors import CORSMiddleware
import math
import time
import random
from functools import lru_cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game constants
ROWS = 6
COLS = 7
WIN_LENGTH = 4
MAX_DEPTH = 8  # Increased for better lookahead
TIME_LIMIT = 3  # Seconds to ensure we respond within 5s limit

# Define key pattern values for faster evaluation
FOUR_IN_ROW = 100000000
THREE_IN_ROW = 1000
TWO_IN_ROW = 100
BLOCK_THREE = 1200
BLOCK_TWO = 100

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    evaluation: Optional[int] = None
    depth: Optional[int] = None
    execution_time: Optional[float] = None

class Connect4AI:
    # Cache for lines and transposition table
    _cached_winning_lines: Optional[List[List[Tuple[int, int]]]] = None
    _transposition_table: Dict[str, Tuple[int, int, int, bool]] = {}  # hash -> (score, depth, move, is_exact)
    
    @staticmethod
    def board_hash(board: List[List[int]]) -> str:
        """Create a unique string representation of the board for caching"""
        return ''.join(''.join(str(cell) for cell in row) for row in board)
    
    @staticmethod
    def get_winning_lines() -> List[List[Tuple[int, int]]]:
        """Get all potential winning lines (cached)"""
        if Connect4AI._cached_winning_lines is not None:
            return Connect4AI._cached_winning_lines

        lines = []
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                lines.append([(r, c + i) for i in range(WIN_LENGTH)])
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - WIN_LENGTH + 1):
                lines.append([(r + i, c) for i in range(WIN_LENGTH)])
        # Diagonal down-right
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS - WIN_LENGTH + 1):
                lines.append([(r + i, c + i) for i in range(WIN_LENGTH)])
        # Diagonal up-right
        for r in range(WIN_LENGTH - 1, ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                lines.append([(r - i, c + i) for i in range(WIN_LENGTH)])
        Connect4AI._cached_winning_lines = lines
        return lines

    @staticmethod
    def evaluate_window(window: List[int], player: int) -> int:
        """Evaluate a window of 4 positions"""
        opponent = 3 - player
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        # No score if both players have pieces in the window
        if player_count > 0 and opponent_count > 0:
            return 0
            
        # Player pieces evaluation
        if player_count == 4:
            return FOUR_IN_ROW
        elif player_count == 3 and empty_count == 1:
            return THREE_IN_ROW
        elif player_count == 2 and empty_count == 2:
            return TWO_IN_ROW
        elif player_count == 1 and empty_count == 3:
            return 1
            
        # Opponent pieces evaluation (negative scores)
        if opponent_count == 3 and empty_count == 1:
            return -BLOCK_THREE  # Critical to block
        elif opponent_count == 2 and empty_count == 2:
            return -BLOCK_TWO
        elif opponent_count == 1 and empty_count == 3:
            return -1
            
        return 0

    @staticmethod
    def evaluate_position(board: List[List[int]], player: int) -> int:
        """Comprehensive board evaluation function"""
        score = 0
        center_col = COLS // 2
        
        # Center column control is strategically important
        center_array = [board[r][center_col] for r in range(ROWS)]
        center_player_count = center_array.count(player)
        score += center_player_count * 3
        
        # Evaluate all potential winning lines
        winning_lines = Connect4AI.get_winning_lines()
        for line in winning_lines:
            window = [board[r][c] for r, c in line]
            score += Connect4AI.evaluate_window(window, player)
        
        # Evaluate positional advantages
        for c in range(COLS):
            for r in range(ROWS):
                if board[r][c] == player:
                    # Prefer positions closer to center horizontally
                    distance_from_center = abs(c - center_col)
                    score += (3 - min(3, distance_from_center)) * 5
                    
                    # Prefer lower positions (more stable and enable more builds)
                    score += (ROWS - r) * 3
                    
                    # Check for connected pieces (stronger positions)
                    # Horizontal connections
                    if c > 0 and board[r][c-1] == player:
                        score += 3
                    if c < COLS-1 and board[r][c+1] == player:
                        score += 3
                    
                    # Vertical connections
                    if r < ROWS-1 and board[r+1][c] == player:
                        score += 5  # Vertical builds are strong
                        
                    # Diagonal connections
                    if r > 0 and c > 0 and board[r-1][c-1] == player:
                        score += 2
                    if r > 0 and c < COLS-1 and board[r-1][c+1] == player:
                        score += 2
                    if r < ROWS-1 and c > 0 and board[r+1][c-1] == player:
                        score += 2
                    if r < ROWS-1 and c < COLS-1 and board[r+1][c+1] == player:
                        score += 2
        
        return score

    @staticmethod
    def check_winner(board: List[List[int]]) -> int:
        """Determine if there's a winner on the board"""
        winning_lines = Connect4AI.get_winning_lines()
        for line in winning_lines:
            values = [board[r][c] for r, c in line]
            if values[0] != 0 and values.count(values[0]) == 4:
                return values[0]
        return 0

    @staticmethod
    def detect_threats(board: List[List[int]], player: int) -> List[int]:
        """Detect columns that represent immediate threats (win next move)"""
        threat_columns = []
    
    # Iterate over all columns to check if they are valid for a move
        for col in range(COLS):
            # Skip if column is full
            if board[0][col] != 0:
                continue
                
            # Simulate the move for the current player in this column
            new_board, row = Connect4AI.make_move(board, col, player)
            
            # Check if the current move creates a winning position
            if Connect4AI.check_winner(new_board) == player:
                threat_columns.append(col)
                
        return threat_columns
    @staticmethod
    def is_board_full(board: List[List[int]]) -> bool:
        """Check if board is full (draw)"""
        return all(board[0][c] != 0 for c in range(COLS))

    @staticmethod
    def is_terminal_node(board: List[List[int]]) -> bool:
        """Check if position is terminal (game over)"""
        return Connect4AI.check_winner(board) != 0 or Connect4AI.is_board_full(board)

    @staticmethod
    def get_valid_moves(board: List[List[int]]) -> List[int]:
        """Get valid moves (empty columns)"""
        return [c for c in range(COLS) if board[0][c] == 0]

    @staticmethod
    def get_next_open_row(board: List[List[int]], col: int) -> int:
        """Find next open row in a column"""
        for r in range(ROWS-1, -1, -1):
            if board[r][col] == 0:
                return r
        return -1

    @staticmethod
    def make_move(board: List[List[int]], col: int, player: int) -> Tuple[List[List[int]], int]:
        """Make a move and return new board state and row"""
        row = Connect4AI.get_next_open_row(board, col)
        if row == -1:
            return board, -1
        new_board = [row.copy() for row in board]
        new_board[row][col] = player
        return new_board, row

    @staticmethod
    def order_moves(board: List[List[int]], valid_moves: List[int], player: int) -> List[int]:
        """Order moves for better alpha-beta pruning efficiency"""
        move_scores = []
        
        # First check for immediate winning moves
        winning_moves = Connect4AI.detect_threats(board, player)
        if winning_moves:
            # Prioritize center winning moves
            center_col = COLS // 2
            winning_moves.sort(key=lambda c: abs(c - center_col))
            # Add remaining moves after winning moves
            remaining_moves = [c for c in valid_moves if c not in winning_moves]
            return winning_moves + remaining_moves
        
        # Then check for opponent winning moves to block
        blocking_moves = Connect4AI.detect_threats(board, 3 - player)
        
        # Score moves based on position and potential
        for col in valid_moves:
            score = 0
            
            # Blocking moves get high priority
            if col in blocking_moves:
                score += 1000
                
            # Prefer center and nearby columns
            center_preference = COLS // 2
            score += (4 - min(3, abs(col - center_preference))) * 10
            
            # Check if move leads to good position
            new_board, row = Connect4AI.make_move(board, col, player)
            if row != -1:
                # Small sample evaluation
                score += Connect4AI.evaluate_position(new_board, player) // 1000
                
                # Avoid moves that give opponent winning moves
                opponent_threats = Connect4AI.detect_threats(new_board, 3 - player)
                if opponent_threats:
                    score -= 500
                    
                # Look for two-move win setups
                for next_col in range(COLS):
                    if next_col != col and next_col < COLS:
                        if board[0][next_col] != 0:
                            continue
                        next_board, _ = Connect4AI.make_move(new_board, next_col, player)
                        if Connect4AI.check_winner(next_board) == player:
                            score += 50
            
            move_scores.append((score, col))
            
        # Sort by score descending
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [col for _, col in move_scores]

    @staticmethod
    def negamax_alpha_beta(board: List[List[int]], depth: int, alpha: float, beta: float, 
                          player: int, start_time: float, time_limit: float) -> Tuple[int, Optional[int]]:
        """Negamax with alpha-beta pruning and time management"""
        # Check if we're running out of time
        if time.time() - start_time > time_limit:
            return None, None  # Signal we need to stop search
            
        # Check transposition table
        board_key = Connect4AI.board_hash(board)
        if board_key in Connect4AI._transposition_table:
            score, stored_depth, move, is_exact = Connect4AI._transposition_table[board_key]
            if stored_depth >= depth and is_exact:
                return score, move
        
        # Check for terminal states
        winner = Connect4AI.check_winner(board)
        if winner == player:
            return FOUR_IN_ROW, None
        elif winner == 3 - player:
            return -FOUR_IN_ROW, None
        elif Connect4AI.is_board_full(board):
            return 0, None
        
        # Depth limit reached
        if depth == 0:
            eval_score = Connect4AI.evaluate_position(board, player)
            return eval_score, None
            
        valid_moves = Connect4AI.get_valid_moves(board)
        if not valid_moves:
            return 0, None
            
        # Order moves for better pruning
        ordered_moves = Connect4AI.order_moves(board, valid_moves, player)
        
        best_value = -math.inf
        best_move = ordered_moves[0]  # Default to first move
        
        # Try each move
        for col in ordered_moves:
            new_board, row = Connect4AI.make_move(board, col, player)
            if row == -1:
                continue
                
            # Opponent's turn (negative of opponent's best score)
            value, _ = Connect4AI.negamax_alpha_beta(
                new_board, depth-1, -beta, -alpha, 3 - player, start_time, time_limit
            )
            
            # Check if search was aborted due to time
            if value is None:
                return None, None
                
            value = -value  # Negamax negation
            
            if value > best_value:
                best_value = value
                best_move = col
                
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # Beta cutoff
        
        # Store in transposition table
        Connect4AI._transposition_table[board_key] = (best_value, depth, best_move, True)
        return best_value, best_move

    @staticmethod
    def find_best_move(board: List[List[int]], player: int, valid_moves: List[int]) -> Tuple[int, int, int, float]:
        """Find best move using iterative deepening with time control"""
        if not valid_moves:
            raise ValueError("No valid moves available")
            
        start_time = time.time()
        best_move = valid_moves[0]  # Default to first valid move
        best_score = -math.inf
        max_depth_reached = 0
        
        # Clear transposition table for new search
        Connect4AI._transposition_table.clear()
        
        # First check immediate threats
        
        # Win in one move if possible
        winning_moves = Connect4AI.detect_threats(board, player)
        if winning_moves:
            # Choose center-most winning move
            winning_moves.sort(key=lambda c: abs(c - COLS//2))
            return winning_moves[0], FOUR_IN_ROW, 1, time.time() - start_time
            
        # Block opponent win
        opponent_threats = Connect4AI.detect_threats(board, 3 - player)
        if opponent_threats:
            # Choose center-most blocking move
            opponent_threats.sort(key=lambda c: abs(c - COLS//2))
            return opponent_threats[0], -BLOCK_THREE, 1, time.time() - start_time
        
        # Use iterative deepening to find best move within time limit
        for depth in range(1, MAX_DEPTH + 1):
            try:
                score, move = Connect4AI.negamax_alpha_beta(
                    board, depth, -math.inf, math.inf, player, 
                    start_time, TIME_LIMIT * 0.9  # Use 90% of time limit
                )
                
                # Check if search was aborted due to time
                if score is None:
                    break
                    
                # Update best move if valid
                if move in valid_moves:
                    best_move = move
                    best_score = score
                    max_depth_reached = depth
                    
                # If we found a winning move, no need to search deeper
                if best_score >= FOUR_IN_ROW // 2:
                    break
                    
            except Exception as e:
                print(f"Error at depth {depth}: {str(e)}")
                break
                
            # Break if we're getting close to time limit
            if time.time() - start_time > TIME_LIMIT * 0.8:
                break
        
        return best_move, best_score, max_depth_reached, time.time() - start_time

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        start_time = time.time()
        valid_moves = game_state.valid_moves
        print(game_state.board)
        print(game_state.current_player)
        print(game_state.valid_moves)
        if not valid_moves:
            raise ValueError("No valid moves available")
            
        player = game_state.current_player
        
        # Find best move
        best_move, score, depth, calc_time = Connect4AI.find_best_move(
            game_state.board, player, valid_moves
        )
        
        # Failsafe: Check if returned move is valid
        if best_move not in valid_moves:
            # Try center column first
            center = COLS // 2
            if center in valid_moves:
                best_move = center
            else:
                best_move = valid_moves[0]
        
        return AIResponse(
            move=best_move,
            evaluation=score,
            depth=depth,
            execution_time=calc_time
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Fallback to center or first valid move
        if game_state.valid_moves:
            center = COLS // 2
            if center in game_state.valid_moves:
                return AIResponse(move=center)
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)



