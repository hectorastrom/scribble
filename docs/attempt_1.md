# First attempt: framing this as an OCR problem

File: `src/decode/mouse_to_ocr.py`

Initally I thought this task was one of optical character recognition. Afterall,
we have very robust models to do OCR, and I really wanted to avoid having to
collect my own dataset and train my own model for this task. So, I looked into
some available OCR models.

After a little research, these seemed like my best choices:
1. `ocrmac`, a py library to use Mac's on-device OCR
2. 258M IBM "granite-docling" [OCR model on HuggingFace](https://huggingface.co/ibm-granite/granite-docling-258M)

### ocrmac
Unfortunately, `ocrmac` produced nothing (literally, the annotations return no
guess) on 99% of characters after downsampling my resolution. At full
resolution, it tended to output some nonsense:
```bash
# Complete decoded string (quality=fast): '?•k?????Ik??•kp'
# Complete decoded string (quality=accurate): '?falankay:::::ni calisiitt:.: time...????iii.а наманаBaTE!!• -'
```

### granite-docling
This was much much worse than even the OCR. Here, this model would take ~1 min
to compute its guess per character, then it would just output
`'!!!!!!!!!!!!!!!!'` as its guess for every character.

Frankly, I might not have tried hard enough to make this one work. I was too
turned off by how big and overkill the model was, and yet how bad the results
were. Maybe I configured it wrong...

### The value from this attempt
This attempt wasn't worthless. The code that started this OCR became very useful
for my later attempts in building a robust preprocessing pipeline for the images
and re-familiarizing myself on how to work with images generally in `torch`.

---