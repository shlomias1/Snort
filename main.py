from snort import Snort

def main():
  game = Snort()
  while game.status() == "ongoing":
      print(game)
      print(f"Current player: {game.current_player}")
      try:
          row, col = map(int, input("Enter row and column (separated by space): ").split())
          if not game.make_move(row, col):
              print("Invalid move. Try again.")
      except ValueError:
          print("Invalid input. Please enter two numbers separated by space.")
  print(game)
  print(game.status())

if __name__ == "__main__":
  main()