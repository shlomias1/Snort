# Game board settings
SIZE = 8
NUM_CHANNELS = 6
NUM_BLOKED_CELLS = 3
# Neural network training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50
HIDDEN_LAYER_SIZE = 256
FEATURE_VECTOR_SIZE = SIZE*SIZE*NUM_CHANNELS # (8, 8, 6) = 384
# Parameters for Genroot Games
NUM_GAMES = 10000
NUM_GAMES_FOR_PUCT = 500
# Parameters for MCTS and PUCT
INFO_RETENTION_TIME = 5
MCTS_SIMULATIONS = 2000
PUCT_SIMULATIONS = 2000
EXPLORATION_WEIGHT = 1
RESULT_MAP = {
    "Winner: R": 1,    # Red Victory → 1
    "Winner: B": -1,  # Blue Victory → -1
    "ongoing" : 0
}
CPUCT = 0.5
STARTING_ELO = 1200
ALPHA = 0.3
EPSILON = 0.25
#dirs
DATA_DIR = "data"
LOG_DIR = "snort_logs"
