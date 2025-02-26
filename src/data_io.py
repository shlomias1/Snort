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
        log_msg = "⚠️ No new games to save. Skipping write operation."
        _create_log(log_msg, "Warning", "snort_game_generation_log.txt")
        print(log_msg)
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
    log_msg = f"✅ {num_saved_games} new games saved in append mode. Memory cleared."
    _create_log(log_msg, "Info", "snort_game_generation_log.txt")
    print(log_msg)

def load_data_from_JSON(filename, memmap_file="games_memmap.dat"):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        log_message = "❌ No saved games found or file is empty, generating new games..."
        _create_log(log_message, "Warning")
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
    inputs_memmap = np.memmap(memmap_file, dtype="float16", mode="w+", shape=(total_states, feature_size))
    state_index = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    game_entry = json.loads(line.strip()) 
                    for state_vector in game_entry["encoded_state"]:
                        inputs_memmap[state_index] = state_vector  
                        state_index += 1  
                except json.JSONDecodeError as e:
                    _create_log(f"⚠️ Skipping corrupted line: {e}", "Error")
                    continue
    except Exception as e:
        _create_log(f"⚠️ Error reading JSON file: {e}", "Error")
        return None
    inputs_memmap.flush()
    _create_log(f"✅ Loaded {total_games} games from {filename} into {memmap_file}.", "Info")
    return inputs_memmap 
