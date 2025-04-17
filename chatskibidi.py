from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import math
import time

# Game logic for Connect4
class Connect4:
    def __init__(self):
        self.turn = 0
        self.result = None
        self.terminal = False

    def get_initial_position(self):
        return Position(self.turn)

class Position:
    def __init__(self, turn, mask=0, position=0, num_turns=0):
        self.turn = turn
        self.result = None
        self.terminal = False
        self.num_turns = num_turns
        self.mask = mask
        self.position = position
        self._compute_hash()

    def move(self, loc):
        new_position = self.position ^ self.mask
        new_mask = self.mask | (self.mask + (1 << (loc * 7)))

        new_pos = Position(int(not self.turn), new_mask, new_position, self.num_turns + 1)
        new_pos.game_over()
        return new_pos

    def legal_moves(self):
        bit_moves = []
        for i in range(7):
            col_mask = 0b111111 << 7 * i
            if col_mask != self.mask & col_mask:
                bit_moves.append(i)
        return bit_moves

    def game_over(self):
        connected_4 = self.connected_four_fast()

        if connected_4:
            self.terminal = True
            self.result = 1 if self.turn == 1 else -1
        else:
            self.terminal = False
            self.result = None

        if self.mask == 279258638311359:
            self.terminal = True
            self.result = 0

    def connected_four_fast(self):
        other_position = self.position ^ self.mask

        m = other_position & (other_position >> 7)
        if m & (m >> 14):
            return True
        m = other_position & (other_position >> 6)
        if m & (m >> 12):
            return True
        m = other_position & (other_position >> 1)
        if m & (m >> 2):
            return True
        return False

    def _compute_hash(self):
        position_1 = self.position if self.turn == 0 else self.position ^ self.mask
        self.hash = 2 * hash((position_1, self.mask)) + self.turn

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return isinstance(other, Position) and self.turn == other.turn and self.mask == other.mask and self.position == other.position

# FastAPI application
app = FastAPI()
agent = None
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

class Connect4Agent(BaseModel):
    def __init__(self):
        self.game = Connect4()
        self.strategy = ucb2_agent(3)
        self.pos = self.game.get_initial_position()
        self.board = [[0 for _ in range(7)] for _ in range(6)]

    def create_connect4_position(self, game_state: GameState) -> Connect4:
        """Converts GameState into Connect4 position"""
        connect4 = Connect4()
        pos = connect4.get_initial_position()
        turn = 0 if game_state.current_player == 1 else 1
        moves = []

        # Simplified: Iterate over the board and recreate the moves
        for row in range(5, -1, -1):  # Iterating from bottom row to top
            for col in range(7):
                if game_state.board[row][col] != 0:
                    if row == 0 or game_state.board[row-1][col] == 0:
                        moves.append((col, game_state.board[row][col]))

        for i, (col, player) in enumerate(moves):
            expected_turn = i % 2
            if (player == 1 and expected_turn != 0) or (player == 2 and expected_turn != 1):
                continue
            pos = pos.move(col)

        while pos.turn != turn:
            valid_cols = [col for col in range(7) if game_state.board[0][col] == 0]
            if valid_cols:
                pos = pos.move(valid_cols[0])
            else:
                break

        return pos


# UCB2 agent logic
def get_nodes(initial_pos, time_limit):
    nodes = {}
    nodes[initial_pos] = (0.0, 0.0, {initial_pos: 0})
    start_time = time.time()

    while time.time() - start_time < time_limit:
        leaf_path = get_leaf(nodes, initial_pos)
        leaf = leaf_path[-1]
        _, ni, _ = nodes[leaf]

        if ni > 0 and not leaf.terminal:
            legal_moves = leaf.legal_moves()
            for loc in legal_moves:
                new_pos = leaf.move(loc)
                if new_pos not in nodes:
                    nodes[new_pos] = (0.0, 0.0, {leaf: 0})

            loc = random.choice(legal_moves)
            child_pos = leaf.move(loc)
            reward = 0
            num_runs = 10
            for _ in range(num_runs):
                reward += randomly_play(child_pos)
            w, n, parent_n_dict = nodes[child_pos]
            if leaf not in parent_n_dict:
                parent_n_dict[leaf] = 0
            parent_n_dict[leaf] += 1
            nodes[child_pos] = (w + reward, n + num_runs, parent_n_dict)

        else:
            reward = 0
            num_runs = 10
            for _ in range(num_runs):
                reward += randomly_play(leaf)

        parent = initial_pos
        for position in leaf_path:
            w, n, parent_n_dict = nodes[position]
            parent_n_dict[parent] += num_runs
            nodes[position] = (w + reward, n + num_runs, parent_n_dict)
            parent = position

    return nodes

def ucb2_agent(time_limit):
    def strat(pos):
        nodes = get_nodes(pos, time_limit)
        player = pos.turn
        best_score = float('-inf')
        if player == 1:
            best_score = float('inf')
        next_best_move = None

        for loc in pos.legal_moves():
            next_pos = pos.move(loc)
            if next_pos not in nodes:
                score = 0.0
            else:
                w, n, _ = nodes[next_pos]
                score = w / n if n > 0 else 0.0

            if score < best_score and player == 1:
                best_score = score
                next_best_move = loc
            elif score > best_score and player == 0:
                best_score = score
                next_best_move = loc
        return next_best_move

    return strat

def randomly_play(pos):
    cur_pos = pos
    while not cur_pos.terminal:
        moves = cur_pos.legal_moves()
        loc = random.choice(moves)
        cur_pos = cur_pos.move(loc)
    return float(cur_pos.result)

def get_leaf(nodes, root):
    current_node = root
    path = []
    while True:
        w, ni, _ = nodes[current_node]
        path.append(current_node)
        if ni == 0:
            return path

        legal_moves = current_node.legal_moves()
        next_player = current_node.turn

        best_score = float('-inf')
        if next_player == 1:
            best_score = float('inf')
        next_best_node = None

        for loc in legal_moves:
            result_position = current_node.move(loc)
            if result_position not in nodes:
                return path
            temp_w, temp_ni, temp_parent_n_count = nodes[result_position]
            if current_node not in temp_parent_n_count:
                temp_parent_n_count[current_node] = 0
            if temp_parent_n_count[current_node] == 0:
                path.append(result_position)
                return path

            score = get_score(nodes[current_node][1], temp_parent_n_count[current_node], temp_w / temp_ni, next_player)
            if score < best_score and next_player == 1:
                best_score = score
                next_best_node = result_position
            elif score > best_score and next_player == 0:
                best_score = score
                next_best_node = result_position

        current_node = next_best_node
        if current_node is None:
            return path

def get_score(N, ni, r, player, c=2.0):
    if player == 0: return r + math.sqrt(c * math.log(N) / ni)
    return r - math.sqrt(c * math.log(N) / ni)

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        print("GameState received:", game_state.dict())

        if not game_state.valid_moves:
            raise ValueError("No valid moves available")

        # Ensure valid moves are in the range and are actually legal
        for col in game_state.valid_moves:
            if col < 0 or col >= 7 or game_state.board[0][col] != 0:
                raise ValueError(f"Invalid move: column {col}")

        pos = Connect4Agent().create_connect4_position(game_state)
        print("Connect4 position:", pos.result)

        selected_move = agent(pos)
        print("Selected move:", selected_move)

        if selected_move not in game_state.valid_moves:
            raise ValueError(f"ucb2_agent returned invalid move: {selected_move}")

        return AIResponse(move=selected_move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    agent = ucb2_agent(3)  # Initialize the agent
    uvicorn.run(app, host="0.0.0.0", port=8080)
