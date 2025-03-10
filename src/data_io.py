import json
import numpy as np
import gc
import os
import pandas as pd
from utils import _create_log
import config

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj] 
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj) 
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()} 
    else:
        return obj

def save_data_to_JSON(new_games,filename):        
    if not new_games:
        _create_log("⚠️ No new games to save. Skipping write operation", "Warning", "snort_game_generation_log.txt")
        return
    with open(filename, "a") as f:
        for game_history, status in new_games:
            game_dict = {
                "encoded_state": convert_to_serializable(game_history),
                "status": status
            }
            f.write(json.dumps(game_dict) + "\n")   
    num_saved_games = len(new_games)
    new_games.clear()
    gc.collect() 
    _create_log(f"✅ {num_saved_games} new games saved in append mode. Memory cleared.", "Info", "snort_game_generation_log.txt")

def load_data_from_JSON_using_memfile(filename, memmap_file="snort_games_memmap.dat"):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        _create_log("❌ No saved games found or file is empty, generating new games...", "Warning")
        return None
    total_states = 0
    total_games = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    total_states += len(json.loads(line.strip())["encoded_state"])
                    total_games +=1
                except json.JSONDecodeError as e:
                    _create_log(f"⚠️ Skipping corrupted line: {e}", "Error")
                    continue
    except Exception as e:
        _create_log(f"⚠️ Error opening JSON file: {e}", "Error")
        return None
    feature_size = config.FEATURE_VECTOR_SIZE
    if os.path.exists(memmap_file):
        os.remove(memmap_file)
    inputs_memmap = np.memmap(memmap_file, dtype="float32", mode="w+", shape=(total_states, feature_size))
    state_index = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    game_entry = json.loads(line.strip()) 
                    for state_vector in game_entry["encoded_state"]:
                        flattened_vector = np.array(state_vector).flatten()
                        inputs_memmap[state_index] = flattened_vector  
                        state_index += 1  
                except json.JSONDecodeError as e:
                    _create_log(f"⚠️ Skipping corrupted line: {e}", "Error")
                    continue
    except Exception as e:
        _create_log(f"⚠️ Error reading JSON file: {e}", "Error")
        return None
    inputs_memmap.flush()
    _create_log(f"✅ Loaded {total_states} states from {filename} into {memmap_file}.", "Info")
    return inputs_memmap 

def load_data_from_JSON(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        _create_log("❌ No saved games found or file is empty, generating new games...", "Warning")
        return None, None, None
    total_states = 0
    total_games = 0
    game_data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    game_entry = json.loads(line.strip())
                    game_data.append(game_entry)
                    total_states += len(game_entry["encoded_state"])
                    total_games +=1
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

    _create_log(f"✅ Loaded {total_games} games with {total_states} states from {filename}.", "Info")
    return inputs, policy_labels, value_labels
