from snort import Snort
from puct import PUCTPlayer
from mcts import MCTSPlayer
from game_net import GameNetwork
from training import PreTrain, Train
from utils import Connect_CUDA, _create_log
import config
    
def main():
    msg = ("Choose from the following options:\n"
           " 1 - For a two-player game.\n"
           " 2 - For a game against an MCTS player.\n"
           " 3 - For a game against a PUCT player.\n")
    game_choice = int(input(msg))
    game = Snort()
    
    if game_choice == 2:
        ai_player = MCTSPlayer(simulations=config.MCTS_SIMULATIONS, exploration_weight=config.EXPLORATION_WEIGHT)
    elif game_choice == 3:  
        Connect_CUDA()
        pretrain = PreTrain(num_games=config.NUM_GAMES)
        games_data = pretrain.load_games_data()
        if games_data is None or len(games_data) == 0:
            _create_log("No game data found. Generating new games...", "Info", "snort_game_generation_log.txt")
            pretrain.generate_self_play_games()
            games_data = pretrain.load_games_data()
        if games_data is None or len(games_data) == 0:
            _create_log("No game data generated.", "Error")
            return
        # inputs, policy_labels, value_labels = pretrain.prepare_training_data(games_data)
        inputs, policy_labels, value_labels = games_data
        if inputs is None or policy_labels is None or value_labels is None:
            _create_log("Training data was not prepared correctly.", "Error")
            return
        network = GameNetwork(input_dim=inputs.shape[1], policy_output_dim=len(policy_labels[0]) , value_output_dim=1)
        trainer = Train(network, inputs, policy_labels, value_labels, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        trainer.train()
        network.load_model("trained_snort_game_network.pth")
        ai_player = PUCTPlayer(network, simulations = config.PUCT_SIMULATIONS, cpuct = config.CPUCT)
    else:
        ai_player = None   
    print("🎲 Initial game board:")
    while game.status() == "ongoing":
        print(f"\n🎮 It's {game.current_player}'s turn!")
        print(game)
        if game_choice == 1 or (game.current_player == "R" and game_choice in [2, 3]):
            try:
                row = int(input("🔹 Enter row to place your piece (0-9): "))
                col = int(input("🔹 Enter column to place your piece (0-9): "))
                game.make_move(row, col)
            except ValueError as e:
                print(f"⚠️ Error: {e}")
                continue
        else:
            print("🤖 AI is thinking...")
            try:
                move = ai_player.choose_move(game)
                if move:
                    game.make_move(*move)
                else:
                    print("⚠️ AI failed to find a move.")
                    break
            except Exception as e:
                print(f"⚠️ AI Error: {e}")
                break
        status = game.status()
        if status != "ongoing":
            print(f"🏁 Game Over! {status}")
            break

if __name__ == "__main__":
    main()
