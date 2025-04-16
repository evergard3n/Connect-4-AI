import tkinter as tk
from tkinter import messagebox
from connect4 import Connect4
from mcts import ucb2_agent
import time
import threading

class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4")
        self.game = Connect4()
        self.pos = self.game.get_initial_position()
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.strategy = ucb2_agent(7)
        # self.strategy = lambda pos: mcts_best_move(pos, time_limit=1.0)
        self.first_player = None  # True if computer goes first, False if player goes first
        self.lock = threading.Lock()  # To prevent race conditions

        # Ask who goes first
        self.ask_first_player()

        # Create the game grid
        self.canvas = tk.Canvas(root, width=700, height=600, bg="blue")
        self.canvas.pack(pady=20)
        self.draw_grid()

        # Bind click event to canvas
        self.canvas.bind("<Button-1>", self.on_click)

        # Start computer's turn if it goes first
        if self.first_player:
            self.root.after(500, self.computer_turn)

    def ask_first_player(self):
        response = messagebox.askyesno("Turn Order", "Do you want to go first?")
        self.first_player = not response  # True if computer goes first

    def draw_grid(self):
        self.canvas.delete("all")
        cell_width = 100
        cell_height = 100
        for i in range(6):
            for j in range(7):
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                if self.board[i][j] == 1:  # Computer
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="red")
                elif self.board[i][j] == 2:  # Player
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="yellow")

    def board_move(self, col, turn):
        for i in range(5, -1, -1):
            if self.board[i][col] == 0:
                self.board[i][col] = 1 if turn == 0 else 2
                return i
        return None

    def check_game_over(self):
        if self.pos.terminal:
            if self.pos.winner == 0:
                messagebox.showinfo("Game Over", "Computer wins!")
            elif self.pos.winner == 1:
                messagebox.showinfo("Game Over", "You win!")
            else:
                messagebox.showinfo("Game Over", "It's a draw!")
            self.root.quit()
            return True
        return False

    def computer_turn(self):
        if self.pos.terminal:
            return
        if (self.first_player and self.pos.turn == 0) or (not self.first_player and self.pos.turn == 1):
            with self.lock:
                move = self.strategy(self.pos)
                row = self.board_move(move, self.pos.turn)
                if row is not None:
                    self.pos = self.pos.move(move)
                    self.draw_grid()
                    if self.check_game_over():
                        return
            # Allow player's turn
            self.root.after(500, self.player_turn)
        else:
            self.root.after(100, self.computer_turn)  # Check again soon

    def player_turn(self):
        if self.pos.terminal:
            return
        if (self.first_player and self.pos.turn == 1) or (not self.first_player and self.pos.turn == 0):
            return  # Wait for player's click
        self.root.after(100, self.player_turn)  # Check again soon

    def on_click(self, event):
        if self.pos.terminal:
            return
        if (self.first_player and self.pos.turn == 1) or (not self.first_player and self.pos.turn == 0):
            col = event.x // 100  # Calculate column from click position
            if 0 <= col < 7 and self.board[0][col] == 0:  # Valid move
                with self.lock:
                    row = self.board_move(col, self.pos.turn)
                    if row is not None:
                        self.pos = self.pos.move(col)
                        self.draw_grid()
                        if self.check_game_over():
                            return
                # Trigger computer's turn
                self.root.after(500, self.computer_turn)

def main():
    root = tk.Tk()
    app = Connect4GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()