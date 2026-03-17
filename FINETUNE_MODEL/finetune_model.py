import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## ATTENTION !!!!!! ##
# This version of PAULow does not enable cropping preprocessing #
# Due to convolution layer constraints: #

# PLEASE ENSURE YOUR INPUT IMAGES/MASKS have the same dimensions during training when finetuning, and make sure all of them are the same size #
# Image dimension should be recorded in model name, i.e. 256_1024_filament.pth -> means model trained on images of 256*1024 #

#
#                         width
#                  +-------------------+
#                  |                   |
#        height    |                   |  
#                  |                   |
#                  +-------------------+
#

# Before finetuning starts, this program will prompt you to input the width and height for reminder and for related issues. Thanks! #

import re
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.dataset_loader import SegmentationDataset
from utils.model import AttentionUNet
from clear_temporary_images import clear_images_in_folder
from utils.clear_finetuning_image import clear_finetune


# =========================
# Paths and Hyperparameters
# =========================
processed_images_dir = "utils/training_images"
processed_masks_dir = "utils/training_masks"

model_dir = "USE_MODEL/saved_models"
save_dir = "USE_MODEL/saved_models"

epochs = 100
batch_size = 4
learning_rate = 1e-5
patience = 100

transform = transforms.Compose([transforms.ToTensor()])


# =========================
# Utility Functions
# =========================
def normalize_image(image):
    """Normalize image to [0, 255] based on max pixel value, channel-aware."""
    image_array = np.array(image, dtype=np.float32)
    max_val = image_array.max()
    if max_val > 0:
        image_array = (image_array / max_val) * 255.0
    return Image.fromarray(image_array.astype(np.uint8))


def natural_sort_key(filename):
    """Sort filenames naturally (e.g., image2 before image10)."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]


def save_normalized_frame(im, output_folder, base_name_prefix, counter, ext):
    normalized = normalize_image(im)
    output_name = f"{base_name_prefix}{counter}{ext}"
    output_path = os.path.join(output_folder, output_name)
    normalized.save(output_path)
    return counter + 1


def process_image_file(image_path, output_folder, counter, is_mask=False):
    ext = os.path.splitext(image_path)[1].lower()
    base_name_prefix = "mask" if is_mask else "image"

    try:
        image = Image.open(image_path)
    except IOError:
        print(f"[ERROR] Could not open image {image_path}")
        return counter

    try:
        image.seek(1)
        is_stack = True
    except EOFError:
        is_stack = False
    image.seek(0)

    if is_stack:
        while True:
            try:
                counter = save_normalized_frame(image, output_folder, base_name_prefix, counter, ext)
                image.seek(image.tell() + 1)
            except EOFError:
                break
    else:
        counter = save_normalized_frame(image, output_folder, base_name_prefix, counter, ext)

    return counter


def process_all_images(input_folder="Finetune_Model/INPUT_IMAGES", output_folder=processed_images_dir):
    print("[INFO] Processing Images")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        sys.exit()

    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files_to_process = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)],
        key=natural_sort_key
    )

    if not files_to_process:
        print(f"[ERROR] No valid image files found in '{input_folder}'.")
        sys.exit()

    counter = 1
    for filename in files_to_process:
        image_path = os.path.join(input_folder, filename)
        counter = process_image_file(image_path, output_folder, counter, is_mask=False)

    print("[INFO] Image Processing Complete")


def process_all_masks(input_folder="Finetune_Model/INPUT_MASKS", output_folder=processed_masks_dir):
    print("[INFO] Processing Masks")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        sys.exit()

    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files_to_process = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)],
        key=natural_sort_key
    )

    if not files_to_process:
        print(f"[ERROR] No valid mask files found in '{input_folder}'.")
        sys.exit()

    counter = 1
    for filename in files_to_process:
        mask_path = os.path.join(input_folder, filename)
        counter = process_image_file(mask_path, output_folder, counter, is_mask=True)

    print("[INFO] Mask Processing Complete")


def parse_training_size_from_filename(model_filename):
    base = os.path.splitext(os.path.basename(model_filename))[0]
    parts = base.split("_")

    if len(parts) < 2:
        raise ValueError(
            "Model filename must start with training height and width, "
            "e.g. 256_1024_filament.pth"
        )

    try:
        train_h = int(parts[0])
        train_w = int(parts[1])
    except ValueError:
        raise ValueError(
            "Could not parse training height/width from model filename. "
            "Expected format like 256_1024_filament.pth"
        )

    if train_h <= 0 or train_w <= 0:
        raise ValueError(f"Parsed invalid training size: {train_h} x {train_w}")

    if train_h % 8 != 0 or train_w % 8 != 0:
        raise ValueError(
            f"Parsed training size {train_h} x {train_w}, but both must be multiples of 8."
        )

    return train_h, train_w


def infer_input_channels_from_state_dict(state_dict):
    candidate_keys = [k for k in state_dict.keys() if "conv1" in k and "weight" in k]
    if not candidate_keys:
        raise ValueError("Could not infer input channels from checkpoint.")
    first_layer_key = candidate_keys[0]
    return state_dict[first_layer_key].shape[1]


def save_sample_images(fold, epoch, images, masks, outputs, model_name, sample_root="utlis/Sample_Images"):
    fold_dir = os.path.join(sample_root, model_name, f"fold{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    for i in range(min(3, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy().transpose(1, 2, 0)
        output = outputs[i].cpu().numpy().transpose(1, 2, 0)

        h, w = img.shape[:2]

        if h >= w:
            fig, axes = plt.subplots(3, 1, figsize=(5, 12))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes = axes.flatten()

        if img.shape[2] == 1:
            axes[0].imshow(img.squeeze(), cmap="viridis")
        else:
            axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(mask.squeeze(), cmap="viridis")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(output.squeeze(), cmap="viridis")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f"fold{fold + 1}_epoch{epoch + 1}_sample{i + 1}.png"))
        plt.close()


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.epochs_without_improvement = 0

    def check_early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience


def verify_finetune_dataset(images_dir, masks_dir, expected_h, expected_w):
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)],
        key=natural_sort_key
    )
    mask_files = sorted(
        [f for f in os.listdir(masks_dir) if f.lower().endswith(valid_exts)],
        key=natural_sort_key
    )

    if not image_files:
        print("[ERROR] No processed images found for fine-tuning.")
        sys.exit()

    if not mask_files:
        print("[ERROR] No processed masks found for fine-tuning.")
        sys.exit()

    if len(image_files) != len(mask_files):
        print(f"[ERROR] Number of images ({len(image_files)}) and masks ({len(mask_files)}) do not match.")
        sys.exit()

    reference_size = None

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[ERROR] Failed to load image: {img_file}")
            sys.exit()

        if mask is None:
            print(f"[ERROR] Failed to load mask: {mask_file}")
            sys.exit()

        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if (img_h, img_w) != (mask_h, mask_w):
            print(f"[ERROR] Image and mask size mismatch: {img_file} ({img_h}x{img_w}) vs {mask_file} ({mask_h}x{mask_w})")
            sys.exit()

        if img_h % 8 != 0 or img_w % 8 != 0:
            print(f"[ERROR] {img_file} has size {img_h}x{img_w}, which is not divisible by 8.")
            sys.exit()

        if reference_size is None:
            reference_size = (img_h, img_w)
        elif (img_h, img_w) != reference_size:
            print(
                f"[ERROR] Not all fine-tuning samples have the same size. "
                f"Expected {reference_size[0]}x{reference_size[1]}, but found {img_h}x{img_w} in {img_file}."
            )
            sys.exit()

    if reference_size != (expected_h, expected_w):
        print(
            f"[ERROR] Fine-tuning sample size is {reference_size[0]}x{reference_size[1]}, "
            f"but selected model expects {expected_h}x{expected_w} from filename."
        )
        sys.exit()

    print(f"[INFO] Fine-tuning dataset verified: all image/mask pairs are {expected_h} x {expected_w}")


def finetune_model(
    model_class,
    dataset,
    batch_size,
    num_epochs,
    model_path,
    model_name=None,
    save_dir="Use_Model/saved_models",
    patience=5,
    learning_rate=1e-5,
    sample_every=5,
    sample_root="Sample_Images"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset is too small for train/validation split.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device)
    expected_input_channels = infer_input_channels_from_state_dict(checkpoint)

    model_instance = model_class(img_ch=expected_input_channels).to(device)
    model_instance.load_state_dict(checkpoint)
    model_instance.train()

    optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_model_state = None

    with tqdm(total=num_epochs, desc="Fine-tuning", ncols=130, leave=True) as epoch_bar:
        for epoch in range(num_epochs):
            model_instance.train()
            running_train_loss = 0.0

            for images, masks, _ in train_loader:
                images = images.float().to(device)
                masks = masks.float().to(device)

                optimizer.zero_grad()

                outputs = model_instance(images)
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=True
                )

                loss = dice_loss(outputs, masks)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)

            model_instance.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for batch_idx, (images, masks, _) in enumerate(val_loader):
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    outputs = model_instance(images)
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[2:],
                        mode="bilinear",
                        align_corners=True
                    )

                    loss = dice_loss(outputs, masks)
                    running_val_loss += loss.item()

                    if (epoch % sample_every == 0) and (batch_idx == 0):
                        save_sample_images(
                            fold=0,
                            epoch=epoch,
                            images=images.cpu(),
                            masks=masks.cpu(),
                            outputs=torch.sigmoid(outputs).cpu(),
                            model_name=model_name,
                            sample_root=sample_root
                        )

            epoch_val_loss = running_val_loss / len(val_loader)

            epoch_bar.set_description(
                f"Fine-tuning | "
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train {epoch_train_loss:.4f} | "
                f"Val {epoch_val_loss:.4f}"
            )
            epoch_bar.update(1)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model_instance.state_dict().items()
                }

            if early_stopping.check_early_stop(epoch_val_loss):
                tqdm.write(f"[INFO] Early stopping at epoch {epoch + 1}.")
                break

    if model_name is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0] + "_finetuned"

    save_path = os.path.join(save_dir, f"{model_name}.pth")

    if best_model_state is not None:
        torch.save(best_model_state, save_path)
    else:
        torch.save(model_instance.state_dict(), save_path)

    print(f"[INFO] Fine-tuned model saved to: {save_path}")
    return save_path


# =========================
# Device Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using device: {device}")


# =========================
# Reminder input
# =========================
print("\n[INPUT] Reminder: enter the original training image size for reference")

try:
    reminder_height = int(input("Training image height: "))
    reminder_width = int(input("Training image width: "))
except ValueError:
    print("[ERROR] Height and width must be integers.")
    sys.exit()

if reminder_height % 8 != 0 or reminder_width % 8 != 0:
    print("[ERROR] Height and width must be multiples of 8.")
    sys.exit()

print(f"[INFO] Reminder training size entered: {reminder_height} x {reminder_width}")


# =========================
# Model Selection
# =========================
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")], key=natural_sort_key)
if not model_files:
    print("[ERROR] No model files found.")
    sys.exit()

print("\nAvailable models:")
for idx, model_file in enumerate(model_files, 1):
    print(f"{idx}. {model_file}")

choice = input("\n[INPUT] Enter Number to choose model: ").strip()

try:
    choice = int(choice)
    if choice < 1 or choice > len(model_files):
        print("[ERROR] Invalid choice.")
        sys.exit()
except ValueError:
    print("[ERROR] Invalid input.")
    sys.exit()

selected_model_file = model_files[choice - 1]
model_path = os.path.join(model_dir, selected_model_file)

try:
    train_h, train_w = parse_training_size_from_filename(selected_model_file)
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit()

print(f"[INFO] Parsed training size from model filename: {train_h} x {train_w}")

if (reminder_height, reminder_width) != (train_h, train_w):
    print(
        f"[WARNING] Reminder size ({reminder_height} x {reminder_width}) does not match "
        f"model filename size ({train_h} x {train_w}). Using model filename size."
    )

checkpoint = torch.load(model_path, map_location=device)
if not isinstance(checkpoint, dict):
    print("[ERROR] Unsupported checkpoint format.")
    sys.exit()

expected_input_channels = infer_input_channels_from_state_dict(checkpoint)
print(f"[INFO] Model expects input channels: {expected_input_channels}")


# =========================
# Preprocessing
# =========================
process_all_images()
process_all_masks()

verify_finetune_dataset(
    images_dir=processed_images_dir,
    masks_dir=processed_masks_dir,
    expected_h=train_h,
    expected_w=train_w
)


# =========================
# Load Dataset
# =========================
dataset = SegmentationDataset(processed_images_dir, processed_masks_dir, transform=transform)

if len(dataset) == 0:
    print("[ERROR] Fine-tuning dataset is empty.")
    sys.exit()

input_channels = dataset.get_input_channels()
print(f"[INFO] Fine-tuning dataset channels: {input_channels}")

if input_channels != expected_input_channels:
    print(
        f"[ERROR] Dataset channels ({input_channels}) do not match model channels "
        f"({expected_input_channels})."
    )
    sys.exit()


# =========================
# Run Fine-tuning
# =========================
finetuned_model_name = selected_model_file.replace(".pth", "_finetuned")

finetune_model(
    model_class=AttentionUNet,
    dataset=dataset,
    batch_size=batch_size,
    num_epochs=epochs,
    model_path=model_path,
    model_name=finetuned_model_name,
    save_dir=save_dir,
    patience=patience,
    learning_rate=learning_rate,
    sample_every=5,
    sample_root="Sample_Images"
)


# =========================
# Cleanup
# =========================
clear_images_in_folder()
clear_finetune()