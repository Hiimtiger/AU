import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import tifffile as tiff
import cv2

from utils.model import AttentionUNet
from utils.clear_use_model_input import clear_images_in_folder

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# =========================
# Paths
# =========================
generated_path = "Use_Model"
input_folder = os.path.join(generated_path, "INPUT_IMAGES")
output_folder = os.path.join(generated_path, "OUTPUT_MASKS")
model_folder = os.path.join(generated_path, "saved_models")

os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(input_folder):
    print(f"[ERROR] Input folder not found: {input_folder}")
    sys.exit()

if not os.path.exists(model_folder):
    print(f"[ERROR] Model folder not found: {model_folder}")
    sys.exit()


# =========================
# Helper: parse train size from model filename
# Example: 512_256_filaments.pth
# =========================
def parse_training_size_from_filename(model_filename):
    base = os.path.splitext(os.path.basename(model_filename))[0]
    parts = base.split("_")

    if len(parts) < 2:
        raise ValueError(
            "Model filename must start with training height and width, "
            "e.g. 512_256_filaments.pth"
        )

    try:
        train_h = int(parts[0])
        train_w = int(parts[1])
    except ValueError:
        raise ValueError(
            "Could not parse training height/width from model filename. "
            "Expected format like 512_256_filaments.pth"
        )

    if train_h <= 0 or train_w <= 0:
        raise ValueError(
            f"Parsed invalid training size: {train_h} x {train_w}"
        )

    if train_h % 8 != 0 or train_w % 8 != 0:
        raise ValueError(
            f"Parsed training size {train_h} x {train_w}, but both must be multiples of 8."
        )

    return train_h, train_w


# =========================
# Helper: infer input channels from checkpoint
# =========================
def infer_input_channels_from_state_dict(state_dict):
    candidate_keys = [k for k in state_dict.keys() if "conv1" in k and "weight" in k]
    if not candidate_keys:
        raise ValueError("Could not infer input channels from checkpoint.")
    first_layer_key = candidate_keys[0]
    return state_dict[first_layer_key].shape[1]


# =========================
# Load model
# =========================
model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".pth")])

if not model_files:
    print("[ERROR] No model files found.")
    sys.exit()

print("\nAvailable models:")
for idx, model_file in enumerate(model_files, 1):
    print(f"{idx}. {model_file}")

choice = input("\n[INPUT] Enter number to choose model: ").strip()

try:
    choice = int(choice)
    if choice < 1 or choice > len(model_files):
        print("[ERROR] Invalid choice.")
        sys.exit()
except ValueError:
    print("[ERROR] Invalid input.")
    sys.exit()

selected_model_file = model_files[choice - 1]
model_path = os.path.join(model_folder, selected_model_file)

try:
    train_h, train_w = parse_training_size_from_filename(selected_model_file)
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit()

print(f"[INFO] Parsed training window from filename: {train_h} x {train_w}")

checkpoint = torch.load(model_path, map_location=device)

# support plain state_dict only
if not isinstance(checkpoint, dict):
    print("[ERROR] Unsupported checkpoint format.")
    sys.exit()

expected_input_channels = infer_input_channels_from_state_dict(checkpoint)
print(f"[INFO] Model expects input channels: {expected_input_channels}")

model = AttentionUNet(img_ch=expected_input_channels).to(device)
model.load_state_dict(checkpoint)
model.eval()


# =========================
# Preprocessing helpers
# =========================
def normalize_image(img):
    """
    Assumes 8-bit input images are the main use case.
    Converts to float32 [0, 1].
    """
    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def adapt_channels(image, expected_channels):
    """
    Make image channel count match the model.
    If too few channels: zero-pad channels.
    If too many channels: truncate extra channels.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if image.shape[-1] < expected_channels:
        pad_channels = expected_channels - image.shape[-1]
        padding = np.zeros(
            (image.shape[0], image.shape[1], pad_channels),
            dtype=image.dtype
        )
        image = np.concatenate([image, padding], axis=-1)

    elif image.shape[-1] > expected_channels:
        image = image[:, :, :expected_channels]

    return image


def load_and_preprocess_image(path):
    """
    Supports:
    - [H, W]
    - [H, W, C]
    - [Z, H, W]
    - [Z, H, W, C]
    """
    img = tiff.imread(path)
    img = np.asarray(img)

    # [H, W]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        img = adapt_channels(img, expected_input_channels)
        img = normalize_image(img)
        return [img], False

    # [H, W, C]
    elif img.ndim == 3 and img.shape[2] <= 4:
        img = adapt_channels(img, expected_input_channels)
        img = normalize_image(img)
        return [img], False

    # [Z, H, W]
    elif img.ndim == 3:
        slices = []
        for z in range(img.shape[0]):
            s = img[z, :, :]
            s = np.expand_dims(s, axis=-1)
            s = adapt_channels(s, expected_input_channels)
            s = normalize_image(s)
            slices.append(s)
        return slices, True

    # [Z, H, W, C]
    elif img.ndim == 4:
        slices = []
        for z in range(img.shape[0]):
            s = adapt_channels(img[z], expected_input_channels)
            s = normalize_image(s)
            slices.append(s)
        return slices, True

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def image_to_tensor(img):
    """
    img: H x W x C, float32
    returns: 1 x C x H x W
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    return tensor


def pad_to_minimum_size(img, min_h, min_w):
    """
    Pad bottom and right so that image is at least min_h x min_w.
    """
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)

    padded = cv2.copyMakeBorder(
        img,
        0, pad_h,
        0, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return padded


def generate_weight_map(h, w):
    """
    Smooth blending window.
    """
    wy = np.hanning(h)
    wx = np.hanning(w)

    if np.all(wy == 0):
        wy = np.ones(h, dtype=np.float32)
    if np.all(wx == 0):
        wx = np.ones(w, dtype=np.float32)

    weight = np.outer(wy, wx).astype(np.float32)
    weight = np.maximum(weight, 1e-6)
    return weight


# =========================
# Sliding-window inference
# =========================
def process_single_image(image, window_h, window_w, stride=1, threshold=0.5):
    """
    Sliding-window inference on arbitrary-size image using the training size
    parsed from the model filename.

    image: H x W x C
    output: H x W uint8 mask
    """
    original_h, original_w = image.shape[:2]

    image = pad_to_minimum_size(image, window_h, window_w)
    padded_h, padded_w = image.shape[:2]

    ys = list(range(0, padded_h - window_h + 1, stride))
    xs = list(range(0, padded_w - window_w + 1, stride))

    if ys[-1] != padded_h - window_h:
        ys.append(padded_h - window_h)
    if xs[-1] != padded_w - window_w:
        xs.append(padded_w - window_w)

    accum_prob = np.zeros((padded_h, padded_w), dtype=np.float32)
    accum_weight = np.zeros((padded_h, padded_w), dtype=np.float32)

    weight_map = generate_weight_map(window_h, window_w)

    total_patches = len(ys) * len(xs)
    processed = 0

    for y in ys:
        for x in xs:
            patch = image[y:y + window_h, x:x + window_w]
            patch_tensor = image_to_tensor(patch)

            with torch.no_grad():
                pred = torch.sigmoid(model(patch_tensor))

            pred_np = pred.squeeze().cpu().numpy().astype(np.float32)

            accum_prob[y:y + window_h, x:x + window_w] += pred_np * weight_map
            accum_weight[y:y + window_h, x:x + window_w] += weight_map

            processed += 1
            if processed % 5000 == 0:
                print(f"[INFO] Processed {processed}/{total_patches} patches")

    merged_prob = np.divide(
        accum_prob,
        accum_weight,
        out=np.zeros_like(accum_prob),
        where=accum_weight > 0
    )

    merged_mask = (merged_prob > threshold).astype(np.uint8) * 255
    merged_mask = merged_mask[:original_h, :original_w]

    return merged_mask


# =========================
# Run inference
# =========================
valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

input_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)])

if not input_files:
    print("[ERROR] No valid input images found.")
    sys.exit()

for fname in input_files:
    full_path = os.path.join(input_folder, fname)

    try:
        images, is_stack = load_and_preprocess_image(full_path)

        if is_stack:
            print(f"\n--> Processing stack: {fname}")
            output_stack = []

            for idx, img in enumerate(images):
                print(f"[INFO] Slice {idx + 1}/{len(images)}")
                result = process_single_image(
                    image=img,
                    window_h=train_h,
                    window_w=train_w,
                    stride=1,
                    threshold=0.5
                )
                output_stack.append(result)

            output_stack = np.stack(output_stack, axis=0)
            save_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.tif")
            tiff.imwrite(save_path, output_stack)
            print(f"[INFO] Saved: {save_path}")

        else:
            print(f"\n--> Processing single image: {fname}")
            result = process_single_image(
                image=images[0],
                window_h=train_h,
                window_w=train_w,
                stride=1,
                threshold=0.5
            )

            save_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.tif")
            tiff.imwrite(save_path, result)
            print(f"[INFO] Saved: {save_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")

print("\n[INFO] Mask generation complete.")
clear_images_in_folder()