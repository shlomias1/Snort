import copy
import numpy as np
import random
import config

class Snort:
    def __init__(self, size=config.SIZE, blocked_cells=None):
        """ Initialize the game board """
        self.size = size
        self.board = np.full((size, size), None)  # None means empty
        self.current_player = "R"  # Red starts
        self.blocked_cells = self.generate_blocked_cells(blocked_cells)
        for cell in self.blocked_cells:
            self.board[cell] = "X"

    def generate_blocked_cells(self, blocked_cells):
        """ Generate random blocked cells if not provided """
        if blocked_cells is None:
            blocked_cells = set()
            while len(blocked_cells) < config.NUM_BLOKED_CELLS:
                random_cell = (random.randint(0, self.size-1), random.randint(0, self.size-1))
                if random_cell not in blocked_cells:
                    blocked_cells.add(random_cell)
        else:
            blocked_cells = set(blocked_cells)
        return blocked_cells

    def switch_turn(self):
        """ Switch the current player's turn """
        self.current_player = "B" if self.current_player == "R" else "R"

    def make_move(self, row, col):
        """ Apply a move to the game state """
        if self.is_legal_move(row, col):
            self.board[row, col] = self.current_player
            self.switch_turn()
            return True
        return False

    def unmake_move(self, row, col):
        """ Reverse a move """
        if self.board[row, col] in ["R", "B"]:
            self.board[row, col] = None
            self.switch_turn()

    def clone(self):
        """ Create a deep copy of the game state """
        return copy.deepcopy(self)

    def encode(self):
        """ Encode the game state as a binary vector including player turn """
        encoding = np.zeros((self.size, self.size, 4))  # 4 channels: R, B, blocked, current player
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == "R":
                    encoding[r, c, 0] = 1
                elif self.board[r, c] == "B":
                    encoding[r, c, 1] = 1
                elif (r, c) in self.blocked_cells:
                    encoding[r, c, 2] = 1
        encoding[:, :, 3] = 1 if self.current_player == "R" else 0
        return encoding

    def decode(self, action_index):
        """ Translate an action index into a move """
        row = action_index // self.size
        col = action_index % self.size
        return row, col

    def status(self):
        """ Return the game result or indicate that the game is ongoing """
        if not self.legal_moves():
            return "Winner: " + ("B" if self.current_player == "R" else "R")
        return "ongoing"

    def is_legal_move(self, row, col):
        """ Check if a move is legal """
        if row < 0 or col < 0 or row >= self.size or col >= self.size:
            return False
        if self.board[row, col] is not None:
            return False
        if any(
            0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == ("B" if self.current_player == "R" else "R")
            for r, c in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        ):
            return False
        return True

    def legal_moves(self):
        """ Return a list of all legal moves """
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.is_legal_move(r, c)]

    def __str__(self):
        board_representation = "  " + " ".join(str(i) for i in range(self.size)) + "\n"
        for i, row in enumerate(self.board):
            board_representation += str(i) + " " + " ".join([cell if cell else "-" for cell in row]) + "\n"
        return board_representation
