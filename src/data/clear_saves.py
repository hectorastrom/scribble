# @Time    : 2026-01-14 23:25
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : clear_saves.py

# script to clear the last X trials from a given range of letters (for when i
# mess up data collection)

import os
from src.data.utils import (
    build_inverse_char_map,
    folder_name_to_char,
    delete_sample,
    RAW_MOUSE_DATA_DIR,
)


def main():
    last_x = int(input("How many of the last X trials do you want to clear: "))
    start_letter = input("Starting at letter: ")
    end_letter = input("Ending at letter: ")

    y_n = input(
        f"Confirm: remove the last {last_x} trials from {start_letter} to {end_letter} (y/n): "
    )
    if y_n.lower() != "y":
        print("Stopping")
        exit(0)

    inv_map = build_inverse_char_map()
    start_idx = inv_map[start_letter]
    end_idx = inv_map[end_letter]

    char_idx_set = set(range(start_idx, end_idx + 1))
    for directory in os.listdir(RAW_MOUSE_DATA_DIR):
        char = folder_name_to_char(directory)
        if char not in inv_map:
            continue

        char_idx = inv_map[char]
        if char_idx not in char_idx_set:
            continue

        complete_path = os.path.join(RAW_MOUSE_DATA_DIR, directory)
        
        # get all trial files sorted by modification time (newest first)
        trial_files = []
        for filename in os.listdir(complete_path):
            if "trial" not in filename or not filename.endswith(".csv"):
                continue
            filepath = os.path.join(complete_path, filename)
            trial_files.append((filepath, os.path.getmtime(filepath)))
        
        # sort by modification time, newest first
        trial_files.sort(key=lambda x: x[1], reverse=True)
        
        # remove the last_x most recent files
        for filepath, _ in trial_files[:last_x]:
            filename = os.path.basename(filepath)
            delete_sample(char, filename)
            print(f"removed {filepath}")


if __name__ == "__main__":
    main()
