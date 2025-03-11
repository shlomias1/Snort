# **Snort AI - Alpha Zero**

## **Overview**
This project implements an AI player for the **Snort** game using **Monte Carlo Tree Search (MCTS)** and **Predictor Upper Confidence Tree (PUCT)** algorithms. The AI learns from **self-play**, optimizing move selection and strategy through reinforcement learning.

The project consists of:
- A **game engine** that simulates Snort.
- AI players using **MCTS** and **PUCT** for decision-making.
- A **deep learning model** for policy and value approximation.
- **Self-play training** with an ELO rating system for performance tracking.

---

## **Installation & Setup**
### **Requirements**
- Python 3.8+
- PyTorch
- NumPy
- Pandas

### **Install Dependencies**
```bash
pip install torch numpy pandas
```

### **Clone Repository**
```bash
git clone https://github.com/shlomias1/Snort.git
cd Snort
```

---

## **Project Structure**
```
model/
│── trained_snort_game_network.pth   # The model created from training on the game data
src/
│── data_io.py         # Handles game data storage (JSON)
│── config.py          # Configuration settings
│── game_net.py        # Neural network model for policy & value estimation
│── main.py            # Entry point for training & evaluation
│── mcts.py            # Monte Carlo Tree Search implementation
│── puct.py            # Predictor Upper Confidence Tree algorithm
│── snort.py           # Game engine (rules, board representation)
│── training.py        # Training loop for AI models
│── utils.py           # Utility functions (logging, debugging)
│── README.md          # Project documentation
│── requirements.txt   # Libraries that need to be installed

```

---

## **How to Run**
### **Start Training**
To train the AI with self-play:
```bash
python main.py
```

This will:
1. Initialize the AI model.
2. Simulate self-play games.
3. Update the model using reinforcement learning.
4. Track performance using the **ELO** system.

### **Play Against AI**
Modify `main.py` to allow user input for moves and run:
```bash
python main.py
```

---

## **Technical Details**
### **Game Rules**
Snort is a combinatorial game played on a grid where:
- **Red (R) and Blue (B)** players alternate turns.
- Pieces cannot be placed adjacent to an opponent's piece.
- The player who **cannot move loses**.

### **AI Players**
#### **Monte Carlo Tree Search (MCTS)**
- Simulates games randomly from a given state.
- Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
- Backpropagates results to improve decision-making.

#### **PUCT (Predictor Upper Confidence Tree)**
- Uses a **neural network** for policy and value estimation.
- Expands promising moves first.
- Incorporates **Dirichlet noise** for exploration.

### **Neural Network**
- **Input**: Encoded board state.
- **Output**:
  - **Policy Head** (Move probabilities)
  - **Value Head** (Win probability)
- Trained using **self-play data**.

---

## **Training Process**
1. **Generate Self-Play Games** (`PreTrain` class in `training.py`).
2. **Store Game States** (JSON format in `data_io.py`).
3. **Train Neural Network**:
   - **Policy Loss** (Cross-entropy with MCTS visit counts).
   - **Value Loss** (Mean Squared Error).
4. **ELO Rating Update**: Evaluates model improvements.

### **Training Parameters** (from `config.py`):
- `LEARNING_RATE = 0.001`
- `EPOCHS = 50`
- `PUCT_SIMULATIONS = 700`
- `STARTING_ELO = 1200`
- `ALPHA = 0.3` (Dirichlet noise)

---

## **Future Improvements**
- **Add human vs AI mode**.
- **Optimize neural network architecture** (e.g., use CNN or Transformer).
- **Enhance MCTS rollout strategy**.
- **Improve board state encoding** for efficiency.

---

## **Contributing**
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## **License**
MIT License.

---

## **Author**
Developed by **Shlomi Assayag**.
