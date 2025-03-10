import math
import torch
import config
import numpy as np
import random
from utils import _create_log

class PUCTNode:
    def __init__(self, state, parent=None, move=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}  # Dictionary of move -> child node
        self.P = prior_prob  # Prior probability from policy network
        self.Q = 0  # Average value of this node
        self.N = 0  # Visit count

    def select_child(self, cpuct):
        """Select a child node using the PUCT formula"""
        best_score = -float("inf")
        best_child = None
        sqrt_N_parent = math.sqrt(self.N + 1)  # Avoid division by zero
        for move, child in self.children.items():
            # Compute PUCT score
            U = child.Q + cpuct * child.P * (sqrt_N_parent / (1 + child.N))
            if U > best_score:
                best_score = U
                best_child = child
        return best_child

    def expand(self, game, policy_probs):
        """
        Expand the node by adding child nodes for legal moves.
        :param game: The current game state.
        :param policy_probs: Dictionary of move -> probability from policy network.
        """
        legal_moves = game.legal_moves()
        for idx, move in enumerate(legal_moves):
            if move not in self.children:
                prior_prob = policy_probs[idx]
                new_state = game.clone()
                new_state.make_move(*move)
                self.children[move] = PUCTNode(new_state, parent=self, move=move, prior_prob=prior_prob)

    def update(self, value):
        """
        Update Q-value using incremental averaging.
        :param value: Value estimate (-1 to 1).
        """
        self.N += 1
        self.Q += (value - self.Q) / self.N  # Running average of Q

    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self):
        """Return the child with the highest visit count (N)."""
        return max(self.children.values(), key=lambda c: c.N, default=None)

class PUCTPlayer:
    def __init__(self, network, simulations = 700, cpuct = config.CPUCT):
        self.network = network
        self.simulations = simulations
        self.cpuct = cpuct

    def choose_move(self, game):
        """Perform PUCT search and choose the best move"""
        root = PUCTNode(game.clone())
        state_tensor = torch.tensor(game.encode().flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy_probs, value_estimate = self.network(state_tensor)
        alpha = config.ALPHA 
        epsilon = config.EPSILON
        dirichlet_noise = np.random.dirichlet([alpha] * len(policy_probs))
        policy_probs = (1 - epsilon) * policy_probs + epsilon * dirichlet_noise
        policy_probs = policy_probs.squeeze(0).detach().cpu().numpy()
        legal_moves = game.legal_moves()
        legal_moves_mask = np.zeros(len(policy_probs))
        for idx, move in enumerate(legal_moves):
            legal_moves_mask[idx] = policy_probs[idx]
        legal_moves_mask /= np.sum(legal_moves_mask) if np.sum(legal_moves_mask) > 0 else np.ones(len(legal_moves)) / len(legal_moves)
        root.expand(game, legal_moves_mask)
        root.update(value_estimate.item())
        for _ in range(self.simulations):
            node = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)
        return root.best_child().move if root.best_child() else random.choice(legal_moves)

    def _select(self, node):
        """Traverse the tree using PUCT selection until reaching a leaf."""
        while node.children and node.is_fully_expanded():
            node = node.select_child(self.cpuct)
        return node if node else None

    def _evaluate(self, node):
        """ Evaluate a leaf node using the neural network. return: Value estimate (-1 to 1) """
        state_tensor = torch.tensor(node.state.encode().flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value_estimate = self.network(state_tensor)
        return value_estimate.item()

    def _backpropagate(self, node, value):
        """ Backpropagate the value estimate up the tree """
        while node:
            node.update(value)
            value = -value
            node = node.parent
