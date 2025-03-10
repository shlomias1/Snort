import random
import math
from utils import _create_log
import config

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = move

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            return None
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                continue
            uct_value = (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

class MCTSPlayer:
    def __init__(self, simulations=700, exploration_weight=1.41):
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def choose_move(self, game):
        root = MCTSNode(game.clone())
        for _ in range(self.simulations):
            node = self._select(root)
            if not node:
                continue
            result = self._simulate(node.state)
            self._backpropagate(node, result)
        best_child = root.best_child(self.exploration_weight)
        return best_child.move if best_child else None

    def _select(self, node):
        while node.children and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                return node
            node = best_child
        return self._expand(node)

    def _expand(self, node):
        legal_moves = node.state.legal_moves()
        tried_moves = {child.move for child in node.children}
        for move in legal_moves:
            if move not in tried_moves:
                new_state = node.state.clone()
                new_state.make_move(*move)
                child = MCTSNode(new_state, parent=node, move=move)
                node.add_child(child)
                return child
        return None

    def _simulate(self, game):   
        legal_moves_R = set(game.legal_moves())  
        legal_moves_B = set(game.legal_moves())  
        while game.status() == "ongoing":
            legal_moves = legal_moves_R if game.current_player == "R" else legal_moves_B
            if not legal_moves:
                return game.status()
            move = random.choice(tuple(legal_moves))  
            legal_moves.discard(move)
            game.make_move(*move)
            if game.current_player == "B":
                self._update_legal_moves_set(legal_moves_R, move)
            else:
                self._update_legal_moves_set(legal_moves_B, move)
        return game.status()

    def _update_legal_moves_set(self, legal_moves_set, move):
        row, col = move
        legal_moves_set.discard(move) 
        adjacent_positions = [
            (row - 1, col), (row + 1, col), 
            (row, col - 1), (row, col + 1)
        ]
        for pos in adjacent_positions:
            legal_moves_set.discard(pos)

    def _backpropagate(self, node, result):
        result_value = config.RESULT_MAP.get(result, 0)
        while node:
            node.visits += 1
            if ": " in result:
                result = result.split(": ")[1] 
            if result == "R":
                node.wins += result_value
            else:
                node.wins -= result_value
            result_value = -result_value
            node = node.parent
