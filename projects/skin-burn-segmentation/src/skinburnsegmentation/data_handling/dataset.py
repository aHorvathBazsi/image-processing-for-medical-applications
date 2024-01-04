import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from skinburnsegmentation.data_handling.utils import create_image_mask_mapping, get_files_from_dir
from skinburnsegmentation.constants import (
    image_filename_prefix,
    image_filename_suffix,
    mask_filename_prefix,
)
import random

def get_class_id_from_mask_filename(mask_filename: Path) -> int:
    return mask_filename.stem[-1]

class CustomDataset(Dataset):
    def __init__(self, dataset_dir: Path, image_transform=None, mask_transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = Path(dataset_dir) / "images"
        self.mask_dir = Path(dataset_dir) / "masks"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Could not find images or masks directory in {dataset_dir}")
        self.image_mask_mapping = self.get_image_mask_mapping()

    def get_image_mask_mapping(self):
        image_files = get_files_from_dir(self.image_dir, prefix=image_filename_prefix, suffix=image_filename_suffix)
        mask_files = get_files_from_dir(self.mask_dir, prefix=mask_filename_prefix)
        image_mask_mapping = create_image_mask_mapping(image_files, mask_files)
        return image_mask_mapping

    def __len__(self):
        return len(self.image_mask_mapping)

    def __getitem__(self, idx):
        image_filename, mask_filenames = list(self.image_mask_mapping.items())[idx]

        image = Image.open(self.image_dir / image_filename)
        mask = np.zeros(image.size[::-1], dtype=np.uint8)
        for mask_filename in mask_filenames:
            class_id = get_class_id_from_mask_filename(Path(mask_filename))
            mask_image = Image.open(self.mask_dir / mask_filename)
            mask_array = np.array(mask_image)
            mask[mask_array > 0] = class_id

        return image, mask

    def visualize_data(self, idx: int, overlay: bool = True, save_path: Path | None = None):
        image, mask = self[idx]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Show the image
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # Show the mask
        if overlay:
            ax[1].imshow(image, alpha=0.7)  # Show the image with some transparency
            ax[1].imshow(mask, cmap='jet', alpha=0.5)  # Overlay the mask
            ax[1].set_title("Image with Mask Overlay")
        else:
            ax[1].imshow(mask, cmap='jet')  # Just show the mask
            ax[1].set_title("Mask")
        ax[1].axis("off")

        # Save or show the figure
        if save_path:
            plt.savefig(save_path.name, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":
    dataset_dir = Path.cwd() / "BAMSI_2"
    dataset = CustomDataset(dataset_dir)
    print("HEY")
    image, masks = dataset[0]
    random_list = random.sample(range(0, len(dataset)), 10)
    for idx in random_list:
        dataset.visualize_data(
            idx=idx,
            overlay=False,
            save_path=Path.cwd() / "mask_{}.png".format(idx)
        )  # Save just the mask
        dataset.visualize_data(
            idx=idx,
            overlay=True,
            save_path=Path.cwd() / "image_with_mask_{}.png".format(idx)
        )  # Save mask overlaid on the image
