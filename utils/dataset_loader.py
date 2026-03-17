from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_names = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpeg', '.jpg', '.tif', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def get_input_channels(self):
        # Load the first image to determine the number of channels
        sample_image = np.array(Image.open(os.path.join(self.images_dir, self.image_names[0])))
        if sample_image.ndim == 2:  # Grayscale
            return 1
        return sample_image.shape[2]  # Number of channels (e.g., 3 for RGB, 4 for stacked modalities)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        mask_name = image_name.replace("image", "mask")  
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(image_path)
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # Apply transforms to both image and mask

        return image, mask, image_name