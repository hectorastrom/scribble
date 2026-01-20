# @Time    : 2026-01-12 10:31
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : decipher.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Method: integrate velocities and break figures when there's a gap of
# continuous zero velocities

# 500ms was empirically determined as a good value (a priori it seems too long)
DT_SAMPLING = 15
ZERO_CHUNK_TIME = 500 # ms
ZERO_CHUNK_FRAMES = ZERO_CHUNK_TIME // DT_SAMPLING + 1 # 500ms / 15ms per sample
FILENAME = "mouse_velocities.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=FILENAME)
    args = parser.parse_args()
    
    chunks = integrate_file(args.file)
    print(
        f"\nWith a character spacing set to {ZERO_CHUNK_TIME}ms there are {len(chunks)} identified characters."
    )
    plot_positions(chunks)

###############################
# Integrating velocities
###############################
def integrate_file(filename: str, zero_chunk_frames=ZERO_CHUNK_FRAMES, silent=True):
    """
    Given a file of continuous strokes of multiple characters, integrate_file
    does two primary things:
    1. Integrates velocity of x and y into mouse positions
    2. Chunks the data based on periods of mouse inactivity (zero velocity)
    
    Args:
        filename: Path to CSV containing the mouse velocity data.
        zero_chunk_frames: Number of consecutive frames with zero velocity
            to consider as a chunk separator (new stroke)
        silent: suppresses print statements if True
    
    Returns:
        List of chunks, where each chunk is a tuple (pos_x, pos_y) of
        numpy arrays representing the mouse positions during that stroke.
    """
    # cols are timestamp, vel_x, vel_y
    df = pd.read_csv(filename)
    speed = np.hypot(df["velocity_x"], df["velocity_y"])
    df["stationary"] = speed <= 0.05 # low speed magnitude
    # accumulate pos
    df['pos_x'] = (df['velocity_x'] * DT_SAMPLING / 1000).cumsum().astype('float64')
    df["pos_y"] = (df["velocity_y"] * DT_SAMPLING / 1000).cumsum().astype("float64")
    stationary = df["stationary"].to_numpy(dtype=bool)
    pos_x = df["pos_x"].to_numpy(dtype=np.float32)
    pos_y = df["pos_y"].to_numpy(dtype=np.float32)

    # split into chunks
    chunks = []
    chunk_start_idx = 0
    stationary_frames = 0
    for i, row in enumerate(zip(stationary, pos_x, pos_y)):
        is_stationary, x, y = row
        if is_stationary: 
            stationary_frames += 1
            if stationary_frames == zero_chunk_frames:
                end = i - zero_chunk_frames + 1
                chunks.append((pos_x[chunk_start_idx:end], pos_y[chunk_start_idx:end]))
        else:
            if stationary_frames >= zero_chunk_frames: # sudden change to motion
                chunk_start_idx=i
                if not silent: 
                    print(f"Added a stroke of ~{stationary_frames * 15}ms")
            stationary_frames = 0

    # add the straggler if we don't end on a zero chunk
    if stationary_frames < zero_chunk_frames:
        chunks.append((pos_x[chunk_start_idx:], pos_y[chunk_start_idx:]))

    return chunks

###############################
# Plotting the characters
###############################
def plot_positions(chunks, nrows=4, padding=4):
    if len(chunks) < 4:
        nrows = len(chunks)
    # rows, cols
    ncols = int(np.ceil(len(chunks) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharex=True, sharey=True) # shared dimensions
    plt.suptitle(f"Characters identified with {ZERO_CHUNK_TIME}ms spacing:")

    # find boundary ranges
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for chunk in chunks:
        x, y = chunk
        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))

    print(f"Ranges: x[{int(min_x)} to {int(max_x)}], y[{int(min_y)} to {int(max_y)}]")

    axes[0, 0].set_xlim(int(min_x)-padding, int(max_x)+padding)
    axes[0, 0].set_ylim(-int(max_y)-padding, -int(min_y)+padding) # inverted y

    for i, ax in enumerate(axes.flat): # easy way to idx row col properly
        if i >= len(chunks):
            break

        x, y = chunks[i]
        # high y values mean lower on screen, which is opposite of how its
        # plotted: so we plot -y
        ax.plot(x, -y)

    plt.show()

###############################
# Main
###############################

if __name__ == "__main__":
    main()
