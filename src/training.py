import random
import os
import gc
import json
import torch
import torch.optim as optim
import numpy as np
from snort import Snort
from mcts import MCTSPlayer
from puct import PUCTPlayer
from utils import _create_log, Connect_CUDA
import config
import pandas as pd
import data_io

class PreTrain:
    def __init__(self, num_games=10000, filename="snort_games_data.json", memmap_file="snort_games_memmap.dat"):
        self.num_games = num_games
        self.filename = filename 
        self.memmap_file = memmap_file

    def generate_self_play_games(self, network=None):
        games = []
        for _ in range(self.num_games):
            game = Snort()
            if network:
                red_player = PUCTPlayer(network, simulations=config.PUCT_SIMULATIONS, cpuct=config.CPUCT)
                blue_player = PUCTPlayer(network, simulations=config.PUCT_SIMULATIONS, cpuct=config.CPUCT)
            else:
                red_player = MCTSPlayer(simulations=config.MCTS_SIMULATIONS, exploration_weight=config.EXPLORATION_WEIGHT)
                blue_player = MCTSPlayer(simulations=config.MCTS_SIMULATIONS, exploration_weight=config.EXPLORATION_WEIGHT)
            game_history = []
            while game.status() == "ongoing":
                current_player = red_player if game.current_player == "R" else blue_player
                move = current_player.choose_move(game)
                if move is None:
                    break
                game.make_move(*move)
                game_history.append(game.encode())
            log_message = f"🔄 Generating game {_+1}/{self.num_games} - status: {game.status()}"
            _create_log(log_message, "Info","snort_game_generation_log.txt")
            status = 1 if game.status() == "Winner: R" else -1 if game.status() == "Winner: B" else 0
            games.append((game_history, status))
            if (_+1) % config.INFO_RETENTION_TIME == 0:
                data_io.save_data_to_JSON(games,self.filename)
        if games:
            data_io.save_data_to_JSON(games,self.filename)
        print("✅ Self-play games generated.")
    
    def load_games_data(self):
        if os.path.exists(self.filename):
            return data_io.load_data_from_JSON(self.filename)
  
    def prepare_training_data(self):
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            _create_log(f"❌ JSON file {self.filename} not found or is empty!", "Error")
            return None, None, None
        total_states = 0
        total_games = 0
        game_data = [] 
        try:
            with open(self.filename, "r") as f:
                for line in f:
                    try:
                        game_entry = json.loads(line.strip())
                        game_data.append(game_entry)
                        total_states += len(game_entry["encoded_state"])
                        total_games += 1
                    except json.JSONDecodeError as e:
                        _create_log(f"⚠️ Skipping corrupted line: {e}", "Error")
                        continue
        except Exception as e:
            _create_log(f"⚠️ Error opening JSON file: {e}", "Error")
            return None, None, None
        feature_size = config.FEATURE_VECTOR_SIZE
        inputs = np.zeros((total_states, feature_size), dtype="float32")
        policy_labels = np.zeros((total_states, feature_size), dtype="float32")
        value_labels = np.zeros((total_states,), dtype="float32")
        state_index = 0
        for game_entry in game_data:
            move_counts = game_entry.get("move_counts", {})
            total_visits = sum(move_counts.values()) if move_counts else 1
            for state_vector in game_entry["encoded_state"]:
                flattened_vector = np.array(state_vector).flatten()
                if flattened_vector.shape[0] == feature_size:
                    inputs[state_index] = flattened_vector
                    value_labels[state_index] = game_entry["status"]
                    policy_probs = np.array([move_counts.get(str(i), 0) / total_visits for i in range(feature_size)])
                    policy_labels[state_index] = policy_probs
                    state_index += 1
                else:
                    _create_log(f"⚠️ Skipping invalid state vector of size {flattened_vector.shape[0]}", "Warning")
        _create_log(f"✅ Loaded {total_games} games with {total_states} states from {self.filename}.", "Info")
        return inputs, policy_labels, value_labels
  
class Train:
    def __init__(self, network, inputs, policy_labels, value_labels, epochs=10, batch_size=16):
        self.network = network
        self.inputs = inputs
        self.policy_labels = policy_labels
        self.value_labels = value_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.red_elo = config.STARTING_ELO
        self.blue_elo = config.STARTING_ELO
        self.loaded_game_count = 0 

    def compute_elo(self, RA, RB, result, K=32):
        EA = 1 / (1 + 10 ** ((RB - RA) / 400))
        EB = 1 / (1 + 10 ** ((RA - RB) / 400))
        new_RA = RA + K * (result - EA)
        new_RB = RB + K * ((1 - result) - EB)
        return new_RA, new_RB
    
    def train(self):
        device = Connect_CUDA()
        optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)
        for epoch in range(self.epochs):
            total_correct = 0
            total_samples = 0
            for i in range(0, len(self.inputs), self.batch_size):
                batch_inputs = torch.tensor(self.inputs[i:i+self.batch_size], dtype=torch.float32, device=device, requires_grad=True)
                batch_policy_labels = torch.tensor(self.policy_labels[i:i+self.batch_size], dtype=torch.float32, device=device)
                batch_value_labels = torch.tensor(self.value_labels[i:i+self.batch_size], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                policy_probs, value_estimate = self.network(batch_inputs)
                policy_probs = policy_probs / policy_probs.sum(dim=-1, keepdim=True)
                policy_loss = -torch.sum(batch_policy_labels * torch.nn.functional.log_softmax(policy_probs, dim=-1)) / self.batch_size
                value_loss = torch.mean((batch_value_labels - value_estimate) ** 2)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
                top_k = 5  
                _, predicted_classes = torch.topk(policy_probs, k=top_k, dim=1)
                _, true_classes = torch.topk(batch_policy_labels, k=top_k, dim=1)
                correct = sum([1 if set(pred).intersection(set(true)) else 0 for pred, true in zip(predicted_classes, true_classes)])
                total_correct += correct
                predicted_value = torch.sign(value_estimate).squeeze()  
                true_value = torch.sign(batch_value_labels).squeeze()
                value_correct = (predicted_value == true_value).sum().item()
                total_samples += batch_policy_labels.shape[0]
            accuracy = (total_correct / total_samples) * 100
            value_accuracy = (value_correct / total_samples) * 100
            _create_log(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%, Value Accuracy: {value_accuracy:.2f}%", "Info", "snort_training_log.txt")
            if epoch % 5 == 0:
                pretrain = PreTrain(num_games=config.NUM_GAMES_FOR_PUCT, filename="new_games.json")  
                pretrain.generate_self_play_games(self.network)
                new_games = pretrain.load_games_data()
                os.remove("new_games.json")
                if new_games is not None:
                    new_inputs, new_policy_labels, new_value_labels = new_games
                    self.inputs = np.concatenate((self.inputs, new_inputs))
                    self.policy_labels = np.concatenate((self.policy_labels, new_policy_labels))
                    self.value_labels = np.concatenate((self.value_labels, new_value_labels))
                for i in range(len(new_value_labels)):
                    game_status = new_value_labels[i]
                    if game_status  == 1:  # status = Red wins
                        self.red_elo, self.blue_elo = self.compute_elo(self.red_elo, self.blue_elo, 1)
                    elif game_status  == -1:  # status = Blue wins
                        self.red_elo, self.blue_elo = self.compute_elo(self.red_elo, self.blue_elo, 0)
                    else:  # Draw
                        self.red_elo, self.blue_elo = self.compute_elo(self.red_elo, self.blue_elo, 0.5)
                _create_log(f"🔢 Updated ELO Ratings: Red = {self.red_elo:.2f}, Blue = {self.blue_elo:.2f}", "Info", "snort_training_log.txt")
        self.network.save_model("trained_snort_game_network.pth")
