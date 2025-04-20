import os
import logging
import argparse

def clean_class_folder(class_folder, dry_run=False):
    """Clean a class folder by removing images without 'low', 'bright', or 'normal' prefixes."""
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logging.info(f"No images found in {class_folder}. Skipping.")
        return 0

    removed_count = 0
    for image_file in image_files:
        # Check if filename lacks 'low', 'bright', or 'normal'
        if not any(prefix in image_file for prefix in ['low', 'bright', 'normal']):
            image_path = os.path.join(class_folder, image_file)
            if dry_run:
                logging.info(f"Would remove: {image_path}")
            else:
                try:
                    os.remove(image_path)
                    logging.info(f"Removed: {image_path}")
                    removed_count += 1
                except Exception as e:
                    logging.error(f"Error removing {image_path}: {e}")
        else:
            logging.debug(f"Keeping: {image_file}")

    logging.info(f"Processed {class_folder}: {removed_count} files removed.")
    return removed_count

def main():
    parser = argparse.ArgumentParser(description="Clean Intel dataset by removing original images without 'low', 'bright', or 'normal' prefixes.")
    parser.add_argument('--dry-run', action='store_true', help="List files to be removed without deleting them.")
    args = parser.parse_args()

    base_dir = 'dataset'
    train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
    test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logging.error("Error: dataset/seg_train/seg_train or dataset/seg_test/seg_test not found.")
        return

    total_removed = 0
    for split_dir in [train_dir, test_dir]:
        split_name = 'train' if split_dir == train_dir else 'test'
        logging.info(f"Starting cleaning of {split_name} dataset...")
        for class_name in classes:
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.exists(class_folder):
                logging.warning(f"Class folder {class_folder} not found. Skipping.")
                continue
            removed = clean_class_folder(class_folder, dry_run=args.dry_run)
            total_removed += removed
        logging.info(f"Completed cleaning of {split_name} dataset.")

    logging.info(f"Total files removed: {total_removed}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()