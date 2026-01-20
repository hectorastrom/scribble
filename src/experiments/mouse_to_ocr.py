# @Time    : 2026-01-12 10:34
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : mouse_to_ocr.py

# Goal: Treat the positions in each chunk as images that we pass to an on-device
# OCR model.

from src.decode.decipher import integrate_file
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from ocrmac import ocrmac
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText


def main():
    PLOT_CHAR_IDX = 10 # for debugging; -1 to skip
    OCR_TYPE = "hf_granite" # {'ocrmac', 'hf_granite'}
    QUALITY = "accurate" # {'fast' | 'accurate'} only for ocrmac

    chunks = integrate_file("mouse_velocities.csv")
    complete_str = ""
    for i, (x, y) in enumerate(chunks):
        # x and y are 1D vectors of mouse positions at timestep=idx
        x = t.from_numpy(x.astype(np.float32))
        y = t.from_numpy(y.astype(np.float32))
        img = build_img(x, y)
        img = img.unsqueeze(0).repeat((3, 1, 1)) # add three channels
        if i == PLOT_CHAR_IDX:
            plot_tensor(img)

        # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
        pil_img = to_pil_image(img, mode="RGB") # 8 bit greyscale (only 1 bit needed actually...) 
        char = recognize_char(pil_img, ocr_type=OCR_TYPE, quality=QUALITY)
        complete_str += char
        print("Character:", char)

    print(f"Complete decoded string (quality={QUALITY}): '{complete_str}'")

# ranges from decipher.py: x[-183 to 663], y[-740 to 111]
def build_img(x : t.Tensor, y: t.Tensor, size=900, downsample_size=30, padding=10, dilation=5):
    # normalize ranges to 0->(size or size)
    x_px = ((x - x.min()) / (x.max() - x.min()) * (size - 1)).long()
    y_px = ((y - y.min()) / (y.max() - y.min()) * (size - 1)).long()

    img = t.zeros((size, size))
    img[y_px, x_px] = 1.0

    # problem: plotting this creates a THIN streak of nonzero pixels
    # to resolve this we must dilate the x_px and y_px w/ a convolution
    img = img.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    kernel_size = 2 * dilation + 1 # a square: -dilation...0...dilation
    dilated = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=dilation)

    # then shrink image for efficiency
    # 30 x 30 downsampled image
    downsampled = F.interpolate(dilated, size=(downsample_size, downsample_size), mode="area")
    padded = F.pad(downsampled, (padding, padding, padding, padding), mode="constant", value=0)

    return padded.squeeze() # (H, W)

def plot_tensor(img : t.Tensor):
    img = img.permute(1, 2, 0).numpy() # (H, W, C) for matplotlib
    plt.imshow(img)
    plt.show()

def recognize_char(pil_img, ocr_type="ocrmac", quality : str = "accurate"):
    match ocr_type:
        case "ocrmac":
            # Complete decoded string (quality=fast): '?•k?????Ik??•kp'
            # Complete decoded string (quality=accurate): '?falankay:::::ni calisiitt:.: time...????iii.а наманаBaTE!!• -'
            # lots of unknowns, nothing is right... so this might not work
            # using on-device OCR; credit: https://github.com/straussmaximilian/ocrmac
            annotation = ocrmac.OCR(
                pil_img, recognition_level=quality, language_preference=["en-US"]
            ).recognize()
            if annotation:
                # [(text, confidence, bounding box)]
                return annotation[0][0]
            else:
                return "?"
        case "hf_granite":
            device = t.device("mps" if t.backends.mps.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
            model = AutoModelForImageTextToText.from_pretrained(
                "ibm-granite/granite-docling-258M",
                dtype=t.float16 if device.type == "mps" else t.float32,
            ).to(device)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "This English letter was drawn without lifting the pen. What letter is it? ",
                        },
                    ],
                },
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[pil_img], return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=48,
            )

            # Trim the input tokens from the output
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        case _:
            raise ValueError(f"Unknown ocr_type={ocr_type}")


if __name__ == "__main__":
    main()
