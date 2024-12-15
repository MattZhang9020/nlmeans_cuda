import argparse
import os
import time
import subprocess

import numpy as np

from PIL import Image

PATH_TO_JUDGE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_TO_NOISED_IMAGES = os.path.join(PATH_TO_JUDGE_DIR, '../img/noised_imgs')
PATH_TO_DENOISED_IMAGES = os.path.join(PATH_TO_JUDGE_DIR, '../img/denoised_imgs')
PATH_TO_OUTPUT_IMAGES = os.path.join(PATH_TO_JUDGE_DIR, '../img/output_imgs')

def compare_images(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))
    
    if img1.shape != img2.shape:
        return 0
            
    total_pixels = np.prod(img1.shape)
    
    matching_pixels = np.sum(img1 == img2)
    
    similarity = (matching_pixels / total_pixels) * 100
    
    return similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run judge.')
    
    parser.add_argument('program', help='Path to program.')
    
    args = parser.parse_args()
    
    if not os.path.isdir(PATH_TO_NOISED_IMAGES):
        print('Missing noised_imgs.')
        exit()
    
    if not os.path.isdir(PATH_TO_OUTPUT_IMAGES):
        os.mkdir(PATH_TO_OUTPUT_IMAGES)
    else:
        for img_name in os.listdir(PATH_TO_OUTPUT_IMAGES):
            output_path = os.path.join(PATH_TO_OUTPUT_IMAGES, img_name)
            os.remove(output_path)
            
    total_time = 0.0
    
    img_names = os.listdir(PATH_TO_NOISED_IMAGES)
    img_names.sort(key=lambda fname: int(fname.split('.')[0]))
    
    for img_name in img_names:
        input_path = os.path.join(PATH_TO_NOISED_IMAGES, img_name)
        
        compare_path = os.path.join(PATH_TO_DENOISED_IMAGES, img_name)
        
        output_path = os.path.join(PATH_TO_OUTPUT_IMAGES, img_name)
        
        start_time = time.time()
        subprocess.run(['srun', '-pnvidia', '-N1', '-n1', '-c2', '--gres=gpu:1', args.program, input_path, output_path])
        end_time = time.time()
        
        duration = end_time - start_time
        
        total_time += duration
        
        if not os.path.exists(output_path):
            print(f'[{img_name}] Missing output.')
            continue
        
        similarity = compare_images(compare_path, output_path)
        
        if similarity < 97:
            print(f'[{img_name}] {duration:.02f}s Wrong answer: {similarity:.02f}%')
            continue
        
        print(f'[{img_name}] {duration:.02f}s Correct: {similarity:.02f}%')
    
    print(f'Total time: {total_time:.02f}s')