import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## ATTENTION !!!!!! ##
# This version of PAULow does not enable cropping preprocessing #
# Due to convolution layer constraints, #
#
# PLEASE ENSURE YOUR INPUT IMAGES/MASKS have height & width being multiples of 8 when training, and make sure all of them are the same size #
#
#                         width
#                  +-------------------+
#                  |                   |
#        height    |                   |  
#                  |                   |
#                  +-------------------+
#
# Use model does not limit input sizes due to our sliding window approach #

# Before training starts, this program will prompt you to input the width and height for reminder and for using model related issues. Thanks! #

from sklearn.model_selection import KFold
from torchvision import transforms
from utils.dataset_loader import SegmentationDataset
from utils.model import AttentionUNet
from utils.trainer import train_model
from utils.stack_splitter import process_all_images, process_all_masks
from utils.clear_images import clear_images_in_folder

process_all_images()
process_all_masks()

print("\n[INPUT] Enter the training image size used for the model")

try:
    train_height = int(input("Training image height: "))
    train_width = int(input("Training image width: "))
except ValueError:
    print("[ERROR] Height and width must be integers.")
    sys.exit()

if train_height % 8 != 0 or train_width % 8 != 0:
    print("[ERROR] Height and width must be multiples of 8 for U-Net.")
    sys.exit()

print(f"[INFO] Training image size: {train_height} x {train_width}")


# ============================
# Ask user for model name
# ============================
user_model_name = input("\n[INPUT] Name of the model: ").strip()

if user_model_name == "":
    print("[ERROR] Model name cannot be empty.")
    sys.exit()

model_name = f"{train_height}_{train_width}_{user_model_name}"
print(f"[INFO] Final model prefix: {model_name}\n")

save_dir = "USE_MODEL/saved_models"
print(f"[INFO] Final Model will be saved to: {save_dir}")


# ============================
# Dataset paths
# ============================
images_path = "utils/training_images"
masks_path = "utils/training_masks"


# ============================
# Hyperparameters
# ============================
batch_size = 4
num_epochs = 100
num_folds = 2
learning_rate = 1e-5
patience = 10


# ============================
# Transformations
# ============================
transform = transforms.Compose([
    transforms.ToTensor()
])


# ============================
# Dataset
# ============================
dataset = SegmentationDataset(images_path, masks_path, transform=transform)
input_channels = dataset.get_input_channels()

print(f"[INFO] Detected input channels: {input_channels}")


# ============================
# K-fold setup
# ============================
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


# ============================
# Model factory
# ============================
model_fn = lambda: AttentionUNet(img_ch=input_channels)


# ============================
# Train model
# ============================
fold_train_losses, fold_val_losses, best_model_path = train_model(
    model=model_fn,
    dataset=dataset,
    kf=kf,
    batch_size=batch_size,
    num_epochs=num_epochs,
    model_name=model_name,
    patience=patience,
    learning_rate=learning_rate,
    save_dir=save_dir,
    save_samples=True,
    sample_every=5,
    sample_root="utils/Sample_Images"
)


# ============================
# Cleanup
# ============================
clear_images_in_folder()

print("\n[INFO] Training pipeline complete.")
print(f"[INFO] Best model path: {best_model_path}")