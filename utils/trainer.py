import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================
# Dice Loss
# ===========================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# ===========================
# Early Stopping
# ===========================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.epochs_without_improvement = 0
        self.stopped_early = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.stopped_early = True
            return True

        return False


# ===========================
# Save Sample Images
# if h >= w -> 3 x 1
# if w > h  -> 1 x 3
# ===========================
def save_sample_images(fold, epoch, images, masks, outputs, model_name, sample_root="Sample_Images"):
    fold_dir = os.path.join(sample_root, model_name, f"fold{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    for i in range(min(3, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy().transpose(1, 2, 0)
        output = outputs[i].cpu().numpy().transpose(1, 2, 0)

        h, w = img.shape[:2]

        if h >= w:
            fig, axes = plt.subplots(1, 3, figsize=(5, 12))
        else:
            fig, axes = plt.subplots(3, 1, figsize=(12, 4))

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
        plt.savefig(
            os.path.join(
                fold_dir,
                f"fold{fold + 1}_epoch{epoch + 1}_sample{i + 1}.png"
            )
        )
        plt.close()


# ===========================
# Training Function
# ===========================
def train_model(
    model,
    dataset,
    kf,
    batch_size,
    num_epochs,
    model_name="model",
    patience=5,
    learning_rate=1e-5,
    save_dir="USE_MODEL/saved_models",
    save_samples=True,
    sample_every=5,
    sample_root="Sample_Images"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    fold_train_losses = []
    fold_val_losses = []

    best_fold_val_loss = float("inf")
    best_fold_index = -1
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nStarting Fold {fold + 1}...")

        model_instance = model().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience)

        train_losses = []
        val_losses = []

        with tqdm(total=num_epochs, desc=f"FOLD {fold + 1}", ncols=130, leave=True) as fold_bar:
            for epoch in range(num_epochs):
                # -------------------
                # Training
                # -------------------
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
                train_losses.append(epoch_train_loss)

                # -------------------
                # Validation
                # -------------------
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

                        if save_samples and (epoch % sample_every == 0) and (batch_idx == 0):
                            save_sample_images(
                                fold=fold,
                                epoch=epoch,
                                images=images.cpu(),
                                masks=masks.cpu(),
                                outputs=torch.sigmoid(outputs).cpu(),
                                model_name=model_name,
                                sample_root=sample_root
                            )

                epoch_val_loss = running_val_loss / len(val_loader)
                val_losses.append(epoch_val_loss)

                # update SAME progress bar line instead of printing a new line
                fold_bar.set_description(
                    f"FOLD {fold + 1} | "
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train {epoch_train_loss:.4f} | "
                    f"Val {epoch_val_loss:.4f}"
                )
                fold_bar.update(1)

                if epoch_val_loss < best_fold_val_loss:
                    best_fold_val_loss = epoch_val_loss
                    best_fold_index = fold + 1
                    best_model_state = copy.deepcopy(model_instance.state_dict())

                if early_stopping.check_early_stop(epoch_val_loss):
                    tqdm.write(f"Early stopping at epoch {epoch + 1} for Fold {fold + 1}.")
                    break

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

    best_model_path = None
    if best_model_state is not None:
        best_model_path = os.path.join(
            save_dir,
            f"{model_name}.pth"
        )
        torch.save(best_model_state, best_model_path)
        print(f"\nBest model saved to: {best_model_path}")
        print(f"Best fold: Fold {best_fold_index} | Best Val Loss: {best_fold_val_loss:.4f}")

    print("\nCross-validation complete.\n")

    return fold_train_losses, fold_val_losses, best_model_path