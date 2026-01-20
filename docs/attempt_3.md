## Third attempt: Using timeseries representations
Still in progress, and will update as I go on.

You can view the scripts related to this attempt at `src.ml.architectures.lstm`,
`src.ml.pretrain`.

Currently, LSTM pretraining on my own ~970 mouse samples does not exceed a test
accuracy of 0.16. I suspect there's some bug.

-- 

First fix: 
- Up epochs from 25 to 50 
    - Accuracy continues to climb (and will continue to climb even beyond 50);
      now at 40% test acc (from 16%)
- Include as many trials as possible, and use class weights
    - Previously (can still see this in `src.ml.finetune`) I would discard
      trials from classes with more trials than the min class. This was a very
      destructive way to balance my dataset.
    - Went from 970 to 1100 trials on 1/20/26 when I made this change
    - Didn't actually improve performance markedly on a 50 epoch, 256 batch size
      run (40% -> 39%)
- Up the LR
    - We were climbing steadily at 150 epochs, just very slowly. I increased it by an OOM from
      `1e-3` to `1e-2`

With these changes, we're still capping out around `48%` test accuracy, with
very noisy training.

<img src="a2_lstm_train.png" alt="lstm training over 150 epochs" width=800>
