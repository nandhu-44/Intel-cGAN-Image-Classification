import os
import urllib.request
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
import random
import tempfile
import logging
from cgan_utils import Generator, get_transforms

def load_generator(checkpoint_path, device):
    """Load the trained generator model."""
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/nandhu-44/Light-Level-cGAN/main/models/light_level_cGAN-v0.pth",
            checkpoint_path
        )
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator_state = checkpoint.get('generator_state', checkpoint)
    generator.load_state_dict(generator_state)
    generator.eval()
    return generator

def process_image(generator, input_path, condition, device, blend_alpha=None, enhance=False):
    """Process an image and return the modified PIL image."""
    try:
        input_img = Image.open(input_path).convert('RGB')
    except Exception as e:
        logging.error(f"Error loading image {input_path}: {e}")
        return None
    original_size = input_img.size

    if condition == 0.0:
        return input_img

    transform = get_transforms()
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    model_condition = -1.0 if condition < 0.0 else 1.0
    condition_tensor = torch.tensor([model_condition], dtype=torch.float32).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor, condition_tensor)

    output_tensor = output_tensor.cpu().squeeze(0) * 0.5 + 0.5
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_img = transforms.ToPILImage()(output_tensor)
    output_img = output_img.resize(original_size, Image.Resampling.LANCZOS)

    if blend_alpha is not None and 0.0 < blend_alpha < 1.0:
        output_img = Image.blend(input_img, output_img, blend_alpha)

    if enhance:
        brightness_factor = 1.0 + condition * 0.5
        output_img = ImageEnhance.Brightness(output_img).enhance(brightness_factor)

    return output_img

def process_pred_folder(generator, input_dir, output_dir, conditions, device):
    """Process all images in the prediction folder."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {input_dir}. Exiting.")
        return

    random.shuffle(image_files)
    total_images = len(image_files)
    split_size = total_images // 5
    print(f"Processing {total_images} images, split size: {split_size}")

    for i, image_file in enumerate(image_files):
        if i < split_size:
            condition = conditions[0]  # -0.9
        elif i < 2 * split_size:
            condition = conditions[1]  # -0.5
        elif i < 3 * split_size:
            condition = conditions[2]  # 0.0
        elif i < 4 * split_size:
            condition = conditions[3]  # 0.5
        else:
            condition = conditions[4]  # 0.9

        input_path = os.path.join(input_dir, image_file)
        base_name, ext = os.path.splitext(image_file)

        if condition == 0.0:
            output_filename = f"{base_name}_normal{ext}"
        elif condition < 0.0:
            output_filename = f"{base_name}_low_{condition:.1f}{ext}"
        else:
            output_filename = f"{base_name}_bright_{condition:.1f}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping {output_path}, already exists.")
            continue

        blend_alpha = None
        enhance = False
        alpha_scale = 1.0
        if condition != 0.0:
            base_alpha = (abs(condition) / 1.0) ** 2
            blend_alpha = min(base_alpha * alpha_scale, 0.95)
            enhance = True

        output_img = process_image(generator, input_path, condition, device, blend_alpha, enhance)
        if output_img is None:
            print(f"Failed to process {input_path}. Skipping.")
            continue

        # Save to temp file and rename atomically
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=output_dir) as temp_file:
            temp_path = temp_file.name
        output_img.save(temp_path, quality=95)
        os.replace(temp_path, output_path)
        print(f"Processed {input_path} -> {output_path}")

def main():
    input_dir = os.path.join('dataset', 'seg_pred', 'seg_pred')
    output_dir = os.path.join('dataset', 'seg_pred', 'seg_pred_modified')
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found.")
        return

    model_path = 'models/light_level_cGAN-v0.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = load_generator(model_path, device)

    conditions = [-0.9, -0.5, 0.0, 0.5, 0.9]
    
    print(f"Starting processing of prediction dataset...")
    process_pred_folder(generator, input_dir, output_dir, conditions, device)
    print("Completed processing of prediction dataset.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()