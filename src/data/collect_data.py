# @Time    : 2026-01-13 08:55
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : collect_data.py

"""
Script to collect carefully labeled mouse velocity data for finetuning

0. Setup: build folder structure
  - raw_mouse_data/
      - a/
      - b/
        ...
      - space/
1. Tell user to center mouse and press "space" to begin
2. After start signal, prompt with a character (print) - do NUM_TRIALS=10
   samples per character in a single session
3. Start recording mouse velocities right after move begins to move (following
   start signal) using GlobalInputManager
4. Stop recording as soon as mouse pauses for more than 500ms
5. Save recording in folder corresponding to prompted letter as CSV file
6. Log that character was saved, and that a new trial is ready (press space
   to begin)
7. Continue until all NUM_TRIALS * NUM_CHARACTERS (53) characters have been saved
"""

import os
import time
import argparse
import numpy as np
from colorama import Fore, Back, Style, init as colorama_init
from collections import Counter

from src.data.utils import (
    build_char_map,
    build_inverse_char_map,
    char_to_folder_name,
    generate_trial_filename,
    GlobalInputManager,
    RAW_MOUSE_DATA_DIR as RAW_DIR,
)

# Fix: crash / insane lag after ~40 characters
# 1. enabled permissions (accessibility and input monitoring for VSCode)
# 2. created a persistent, global input manager


SAVE_DELAY_MS_DEFAULT = 500
NUM_TRIALS_DEFAULT = 5
CHAR_MAP = build_char_map()
INV_CHAR_MAP = build_inverse_char_map()
colorama_init(autoreset=True)


# want really clean formatting (so this data labelling isn't so painful...)
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def count_samples_per_char() -> dict[str, int]:
    """
    Count existing training samples for each character by scanning raw_mouse_data folders.
    
    Returns:
        Dict mapping char -> sample count (e.g., {'a': 15, 'B': 3, ' ': 10})
    """
    counts = {}
    for _, char_value in CHAR_MAP.items():
        folder_name = char_to_folder_name(char_value)
        folder_path = os.path.join(RAW_DIR, folder_name)
        
        if os.path.exists(folder_path):
            # count CSV files in the folder
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            counts[char_value] = len(csv_files)
        else:
            counts[char_value] = 0
    
    return counts


def get_prioritized_char_order() -> list[str]:
    """
    Return list of characters sorted by sample count (fewest first).
    
    This prioritizes collecting data for underrepresented characters.
    """
    counts = count_samples_per_char()
    # sort by count ascending, then by char for stable ordering
    sorted_chars = sorted(counts.keys(), key=lambda c: (counts[c], c))
    return sorted_chars


def print_sample_summary(counts: dict[str, int]):
    """Print a summary of current sample counts."""
    print(f"\n{Fore.CYAN}Current sample counts:{Style.RESET_ALL}")
    
    # group by count for cleaner display
    count_groups = Counter(counts.values())
    sorted_counts = sorted(count_groups.keys())
    
    for count in sorted_counts:
        chars_at_count = [c for c, cnt in counts.items() if cnt == count]
        # format chars nicely, showing 'space' for ' '
        formatted_chars = [('space' if c == ' ' else c) for c in sorted(chars_at_count)]
        if len(formatted_chars) <= 10:
            char_str = ', '.join(formatted_chars)
        else:
            char_str = f"{len(formatted_chars)} characters"
        print(f"  {count} samples: {char_str}")
    
    total = sum(counts.values())
    print(f"\n{Fore.CYAN}Total samples: {total}{Style.RESET_ALL}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=int,
        default=NUM_TRIALS_DEFAULT,
        help="Num trials to record per character",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Character to resume from in manual mode (A-Z, a-z, ' ')",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Disable smart prioritization; use sequential a-z order instead",
    )
    args = parser.parse_args()

    global CHAR_MAP
    CHAR_MAP = build_char_map()
    input_manager = GlobalInputManager()
    
    # determine character order based on mode
    if args.manual:
        # manual mode: sequential order (optionally starting from --start)
        start_char_idx = INV_CHAR_MAP[args.start] if args.start else 0
        char_order = [CHAR_MAP[i] for i in range(start_char_idx, len(CHAR_MAP))]
        mode_desc = "Manual mode (sequential a-z order)"
    else:
        # smart mode: prioritize characters with fewest samples
        char_order = get_prioritized_char_order()
        mode_desc = "Smart mode (prioritizing underrepresented characters)"
    
    # show current state
    counts = count_samples_per_char()
    print(f"\n{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}  {mode_desc}  {Style.RESET_ALL}")
    print_sample_summary(counts)

    try:
        setup()
        start_time = time.time()

        num_trials = args.trials
        total_expected = len(char_order) * num_trials
        total_saved = 0

        for char_idx, char_value in enumerate(char_order):
            display_char = "space" if char_value == " " else char_value
            current_count = counts.get(char_value, 0)

            for trial_idx in range(num_trials):
                clear_terminal()

                # show progress info in smart mode
                if not args.manual:
                    print(f"{Fore.CYAN}[{char_idx + 1}/{len(char_order)}] "
                          f"Current samples for '{display_char}': {current_count + trial_idx}{Style.RESET_ALL}")

                # New Letter Prompt:
                if trial_idx == 0:
                    # red header for the very first trial of a new character
                    print(
                        f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}  NEW LETTER: {display_char}  "
                    )
                    print(f"{Fore.RED}{'=' * 30}")
                else:
                    # blue header for subsequent trials of the same letter
                    print(
                        f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT}  NEXT TRIAL: {display_char}  "
                    )
                    print(f"{Fore.BLUE}{'=' * 30}")
                print(
                    f"\n{Style.BRIGHT}Trial {trial_idx + 1}/{num_trials} for '{char_value}'"
                )
                print(f"\n{Style.BRIGHT}Press SPACE to begin recording...")

                # use persistent event listener
                input_manager.wait_for_space()

                # Trial in Progress Prompt:
                clear_terminal()
                print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}  TRIAL IN PROGRESS  ")
                print(f"{Fore.GREEN}{'=' * 30}")
                print(f"\nRecording movement for: {Fore.YELLOW}'{display_char}'")

                folder_name = char_to_folder_name(char_value)
                out_dir = os.path.join(RAW_DIR, folder_name)
                os.makedirs(out_dir, exist_ok=True)
                filename = generate_trial_filename()
                saved_path = os.path.join(out_dir, filename)

                record_one_stroke(
                    saved_path, input_manager, save_delay_ms=SAVE_DELAY_MS_DEFAULT
                )

                total_saved += 1
                # flash of success :)
                print(f"\n{Fore.GREEN}Saved: {saved_path}")
                time.sleep(0.3)

        print(f"\nAll set! {num_trials} files have been saved for each character.")
        end_time = time.time()
        print(
            "Time elapsed: ",
            time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)),
        )
        print(
            f"Your labelling rate: {total_expected / (end_time - start_time):.2f} chars/sec"
        )

    finally:
        # clean input threads
        input_manager.shutdown()


def setup():
    os.makedirs(RAW_DIR, exist_ok=True)
    for _, char_value in CHAR_MAP.items():
        folder_name = char_to_folder_name(char_value)
        os.makedirs(os.path.join(RAW_DIR, folder_name), exist_ok=True)

    print(f"Folder structure ready under: {os.path.abspath(RAW_DIR)}")


def record_one_stroke(
    save_path: str | None,
    input_manager: GlobalInputManager,
    save_delay_ms: int = SAVE_DELAY_MS_DEFAULT,
) -> np.ndarray:
    """
    Record one stroke, starting from first mouse cursor movement until the mouse
    is stationary for `save_delay_ms`.

    Args:
        save_path: Path to save CSV file. If None, data is not saved to disk.
        input_manager: Reference to the global GlobalInputManager instance.
        save_delay_ms: Milliseconds of inactivity before stopping recording.

    Returns:
        np.ndarray of shape (N, 2) with velocity data [velocity_x, velocity_y]
    """
    input_manager.start_recording()

    # wait for first move to begin logging
    input_manager.wait_for_first_move()

    save_delay_s = max(0.0, float(save_delay_ms) / 1000.0)

    # wait for mouse inactivity
    while True:
        time.sleep(0.05)
        if input_manager.get_time_since_last_move() >= save_delay_s:
            break

    # stop recording and get data
    velocities = input_manager.stop_recording()

    # save to CSV if path provided
    if save_path is not None:
        input_manager.save_to_csv(save_path)

    return velocities


if __name__ == "__main__":
    main()
