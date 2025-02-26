from snort import Snort
from puct import PUCTPlayer
from game_net import GameNetwork
from training import PreTrain, Train
from utils import Connect_CUDA, _create_log
import config

def main():
    Connect_CUDA()
    pretrain = PreTrain(num_games=config.NUM_GAMES)
    games_data = pretrain.load_games_data()
    if games_data is None or len(games_data) == 0:
        _create_log("No game data found. Generating new games...", "Info", "snort_game_generation_log.txt")
        pretrain.generate_self_play_games()
        games_data = pretrain.load_games_data()
    if games_data is None or len(games_data) == 0:
        log_message = "No game data generated."
        _create_log(log_message, "Error")
        return
    inputs, policy_labels, value_labels = pretrain.prepare_training_data()
    if inputs is None or policy_labels is None or value_labels is None:
        log_message = "Training data was not prepared correctly."
        _create_log(log_message, "Error")
        print(log_message)
        return
    game = Snort()
    legal_moves = game.legal_moves()
    if not legal_moves:
        log_message = "No legal moves available at the start of the game."
        _create_log(log_message, "Error")
        return
    network = GameNetwork(input_dim=inputs.shape[1], policy_output_dim=len(policy_labels[0]) , value_output_dim=1)
    trainer = Train(network, inputs, policy_labels, value_labels, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    trainer.train()
    network.load_model("trained_game_network.pth")
    puct_player = PUCTPlayer(network, simulations = config.PUCT_SIMULATIONS, cpuct=1.5)
    print("🎲 Initial game board:")
    print(game)
    while not game.game_over:
        print(f"\n🎮 It's {game.turn}'s turn!")
        print(game)
        if game.turn == "R":
            try:
                from_row = int(input("🔹 Enter row to move from (0-9): "))
                from_col = int(input("🔹 Enter column to move from (0-9): "))
                to_row = int(input("🔹 Enter row to move to (0-9): "))
                to_col = int(input("🔹 Enter column to move to (0-9): "))
                game.make_move((from_row, from_col), (to_row, to_col))
            except ValueError as e:
                print(f"⚠️ Error: {e}")
                continue
        else:
            print("🤖 PUCT player is thinking...")
            try:
                move = puct_player.choose_move(game)
                if move:
                    game.make_move(*move)
                    print(f"🤖 PUCT player moved from {move[0]} to {move[1]}")
                else:
                    print("⚠️ PUCT failed to find a move.")
                    break
            except Exception as e:
                print(f"⚠️ PUCT Error: {e}")
                break
        status = game.status()
        if status != "ongoing":
            print(f"🏁 Game Over! {status}")
            break
    print("🏁 Game Over!")

if __name__ == "__main__":
    main()
