import argparse

import numpy as np

from PIL import Image


def compare_images(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))
    
    if img1.shape != img2.shape:
        print('Image size is different.')
        return
            
    total_pixels = np.prod(img1.shape)
    
    matching_pixels = np.sum(img1 == img2)
    
    similarity = (matching_pixels / total_pixels) * 100
    
    print(f'Image similarity: {similarity:.02f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two images for similarity')
    
    parser.add_argument('image1', help='Path to first image')
    parser.add_argument('image2', help='Path to second image')
    
    args = parser.parse_args()
    
    compare_images(args.image1, args.image2)