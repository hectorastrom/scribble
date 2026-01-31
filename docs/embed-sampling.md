# Online Learning 1: NN Embedding Sampling

**Goal**: change sampling from class-based `argmax` to Euclidean nearest neighbor,
treating the output of the finetuned model as a stroke *embedding* rather than a
stroke classification.

**Why**: enables (in theory) arbitrarily many strokes, so users can add custom
strokes. (caveat: embeddings will be most meaningful for strokes seen in training)

## Implementation

Modify `finetune.py` (uses CNN variant of model) to support a new mode:
`nn-sampling`. In `nn-sampling` mode, the output dimension of the CNN is
`embed-dim`, default to `64` (I have idea what number will be good here). 

Write a `knn_forward_pass(model, img, k)` function in `src.ml.utils` that returns the `k` nearest
embeddings to the embedded `img` from `stroke_embeds`, a `(num_strokes, embed_dim)` embedding table learned in the
first offline fine-tuning stage.


```python
# Example knn_forward_pass
def knn_forward_pass(model, img, k=1):
    z = F.normalize(model(img), dim=-1) # normalize across embed dim (B, D)
    E = F.normalize(model.stroke_embeds.weight, dim=-1) # (C, D)
    # squared euclidean distance
    # z.unsqueeze(1): (B, 1, D)
    # E.unsqueeze(0): (1, C, D)
    # z-E : (B, C, D) broadcast
    # sum((z-E)**2, dim=-1): (B, C)
    diff = z.unsqueeze(1) - E.unsqueeze(0)
    dist2 = (dist ** dist).sum(dim=-1)
    vals, idx = torch.topk(dist2, k, largest=False) # nearest neighbors
    return idx
```

**There are three primary lifecycles for this system:**

1. Offline training (fine-tuning after EMNIST pretraining)
    - Update CNN and Embeddings
2. Online training (registering a new stroke)
    - Freeze CNN, Update Embeddings

At the training phase, we're both learning meaningful embeddings `E` for the set
of (currently) 63 strokes we've collected offline data for, **and** learning
model parameters `theta` so that outputs `z` for class `j` are close (in 
Euclidean space) to the embedding `E[j]`.

### Code

The implementation occurs at three stages.

**Updates**:

1. `src.ml.cnn` - update CNN to support `stroke_embeds` embedding table, and use
   negative squared distance as the logits (then loss is cross entropy so class
   with minimal distance is pushed to be the correct one)
    - New args: 
        - `using_embeds: bool`
        - `embed_dim : int`
        - `stroke_embeds : nn.Embedding`
        - `stroke_map: dict[int, str]` 
            - Q: should we use a stroke object, with e.g. `stroke_shorthand`,
              `stroke_filename`? 
            - For starters this is just `src.data.utils.build_char_map()`
2. `src.ml.finetune` - add arg to enable embedding training (offline training)
3. `src.decode.real_time` - check `using_embeds` toggle of CNN to modify stroke
   detection, choosing to use `knn_forward_pass` instead

**New**:

4. `src.ml.online_train` - new script to register a new type of stroke.
   - User titles stroke, provides X examples (default 10), then the embedding table of
   a checkpoint is updated from the average `z` of those 10 samples, stored as a new checkpoint. 