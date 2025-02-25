from snort import Snort
from mcts import MCTSPlayer

def main():
    game = Snort()
    mcts_player = MCTSPlayer()
    while game.status() == "ongoing":
        print(game)
        print(f"Current player: {game.current_player}")
        if game.current_player == "R":
            try:
                row, col = map(int, input("Enter row and column (separated by space): ").split())
                if not game.make_move(row, col):
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter two numbers separated by space.")
        else:
            move = mcts_player.choose_move(game)
            if move:
                game.make_move(*move)
                print(f"MCTS chose move: {move}")
    print(game)
    print(game.status())
