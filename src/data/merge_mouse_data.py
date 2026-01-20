# @Time    : 2026-01-14 11:55
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : merge_mouse_data.py

# Goal: merge another person's collected mouse data with my own

import argparse
import os
from src.data.utils import RAW_MOUSE_DATA_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", type=str,
                        help="Path to the directory containing another person's mouse data")
    parser.add_argument("--to", type=str, default=RAW_MOUSE_DATA_DIR,
                        help="Path to the directory where merged data should be stored")
    args = parser.parse_args()
    
    from_dir = args.__dict__["from"] # .from is used
    to_dir = args.to
    
    assert os.path.exists(from_dir), f"From directory {from_dir} does not exist"
    assert os.path.exists(to_dir), f"To directory {to_dir} does not exist"
    
    ###############################
    # merge data
    ###############################
    
    total_files_moved = 0
    for char in os.listdir(from_dir):
        if not os.path.isdir(os.path.join(from_dir, char)):
            continue
        from_char_dir = os.path.join(from_dir, char)
        to_char_dir = os.path.join(to_dir, char)
        
        # increment trials of FROM data so we don't overwrite
        num_trials = 0
        for filename in os.listdir(to_char_dir):
            if filename.endswith(".csv"):
                num_trials += 1
                
        # insert FROM files starting at num_trials (since we're 0 idx)
        for filename in os.listdir(from_char_dir):
            if filename.endswith(".csv"):
                from_file_path = os.path.join(from_char_dir, filename)
                to_file_path = os.path.join(to_char_dir, f"trial_{num_trials}.csv")
                os.rename(from_file_path, to_file_path)
                num_trials += 1
                total_files_moved += 1

    print(f"Merged data from {total_files_moved} files from {from_dir} into {to_dir}")
    
if __name__ == "__main__":
    main()