# @Time    : 2026-01-14 17:49
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : utils.py

# Contains some logic commonly used across scripts for the models (mainly
# inference)

import torch as t
import time


def forward_pass(model, img : t.Tensor, char_map : dict, log_time : bool = False):
    """
    Run an inference pass with a model given a 4D tensor image input (B, C, H,
    W)
    
    Returns a list of probabilities assigned to each character (length of model
    logits output dim), and the predicted char according to the char_map. If
    log_time is true, the inference time will also be returned.
    """
    assert img.ndim == 4, "Image must have batch and channel dimensions!"
    
    start_time = time.time()
    with t.no_grad():
        input_tensor = img.to(model.device)
        logits = model(input_tensor)
        char_probs = t.softmax(logits, dim=1).squeeze(0) # (num_classes, )
        predicted_idx = t.argmax(logits, dim=1).item()
    end_time = time.time()
    predicted_char = char_map[predicted_idx]
    
    if log_time:
        return char_probs, predicted_char, end_time - start_time
    
    return char_probs, predicted_char,