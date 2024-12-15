import argparse

import numpy as np

from PIL import Image


def add_noise(image_path, save_path, noise_factor=0.5):
    img = Image.open(image_path)
    
    img_array = np.array(img)
    
    noise = np.random.normal(0, 50, img_array.shape)
    
    noisy_img = img_array + noise_factor * noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(noisy_img)
    result.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add noise to image.')
    
    parser.add_argument('input', help='Path to input image')
    parser.add_argument('output', help='Path to output image')
    
    args = parser.parse_args()
    
    add_noise(args.input, args.output)