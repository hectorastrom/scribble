# scribble

## About

**scribble** can decode any cursor movement into one of 53 characters ('a-z',
'A-Z' and space). It does this at a rate exceeding `400 chars/s` running
on an M2 Pro MacBook, by transforming the mouse velocity data into an image of
the character that is then classified by a lightweight (0.3M) CNN.

If you want to read about my progress in building this, challenges I ran into,
and thoughts for the future of the project, read [progress.md](docs/progress.md).

## Usage 

### Try it
This repository contains many, many files. If you just want to get started and
play around with it, run:

```bash
# Try it
uv run -m src.decode.real_time --ckpt best_finetune

# Collect some data
uv run -m src.data.collect_data

# Train it
uv run -m src.ml.finetune --ckpt best_pretrain

# Test it
uv run main.py --ckpt <ckpt_name>
```

## Results

My best finetuned model achieved `92.9%` accuracy on its offline test set, and
feels pretty good online, too.

All detailed results can be found in [attempt 2 results](docs/attempt_2.md)


## Data & Checkpoints
Download my raw_mouse_data (as a base for finetuning) from a public Google Drive folder [HERE](https://drive.google.com/drive/folders/1Jx-_nzswDbaWGWwAS87K4sdrVNIOv87W?usp=sharing).

Download trained model checkpoints — of different architectures and class count heads — at the Drive folder [HERE](https://drive.google.com/drive/folders/180IwTd-S6KBkDkLBXK6kgvWuEV2iRU4M?usp=sharing)

---
