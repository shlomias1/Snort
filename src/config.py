# Game board settings
SIZE = 10
NUM_BLOKED_CELLS = 3
# Neural network training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50
HIDDEN_LAYER_SIZE = 256
FEATURE_VECTOR_SIZE = 400
# Parameters for Genroot Games
NUM_GAMES = 10000
# Parameters for MCTS and PUCT
INFO_RETENTION_TIME = 5
MCTS_SIMULATIONS = 700
PUCT_SIMULATIONS = 700
EXPLORATION_WEIGHT = 1
RESULT_MAP = {
    "Winner: R": 1,    # Red Victory → 1
    "Winner: B": -1  # Blue Victory → -1
}
CPUCT = 1.0
#dirs
DATA_DIR = "data"
LOG_DIR = "snort_logs"
