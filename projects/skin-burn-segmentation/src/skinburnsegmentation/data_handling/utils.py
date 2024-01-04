from pathlib import Path


def get_filenames_from_dir(dir: Path) -> list[Path]:
    return [
        filename for filename in Path(dir).iterdir()
        if filename.is_file()
    ]

def filter_files_with_prefix(files: list[Path], prefix: str) -> list[Path]:
    return [
        file for file in files
        if file.name.startswith(prefix) and file.is_file()
    ]

def filter_files_with_suffix(files: list[Path], suffix: str) -> list[Path]:
    return [
        file for file in files
        if file.name.endswith(suffix) and file.is_file()
    ]

def get_files_from_dir(dir: Path, prefix: str | None, suffix: str | None = None) -> list[Path]:
    files = get_filenames_from_dir(dir)
    if prefix:
        files = filter_files_with_prefix(files, prefix)
    if suffix:
        files = filter_files_with_suffix(files, suffix)
    return files

def create_image_mask_mapping(image_files: list[Path], mask_files: list[Path]) -> dict[str, list[str] | str]:
    # Create a mapping dictionary
    image_filename_masks_mapping = {}

    # Iterate over each image file
    for image_file in image_files:
        # Extract the NUMBER part from the image filename
        image_number = image_file.stem.split('_')[1]

        # Find all mask files that match this NUMBER
        matching_masks = [f for f in mask_files if f.stem.split('_')[1] == image_number]

        # Add to mapping
        image_filename_masks_mapping[image_file.name] = [f.name for f in matching_masks]

    return image_filename_masks_mapping

if __name__ == "__main__":
    image_dir = Path.cwd() / "BAMSI_2" / "images"
    mask_dir = Path.cwd() / "BAMSI_2" / "masks"
    image_files = get_files_from_dir(image_dir, prefix="CROP_", suffix=".jpg")
    mask_files = get_files_from_dir(mask_dir, prefix="MASK_")
    mapping = create_image_mask_mapping(image_files, mask_files)
