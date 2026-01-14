import cv2
import numpy as np
import Augmentor
import os


def augment_images_for_alphabet(input_base_directory, num_samples=1000):
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        input_directory = os.path.join(input_base_directory, letter)
        output_directory = input_directory

        if not os.path.exists(input_directory):
            print(f"Directory {input_directory} does not exist, skipping.")
            continue

        p = Augmentor.Pipeline(input_directory)

        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)

        p.sample(num_samples)

        augmented_subdir = os.path.join(input_directory, 'output')
        augmented_images = os.listdir(augmented_subdir)

        for img_name in augmented_images:
            src = os.path.join(augmented_subdir, img_name)
            dst = os.path.join(output_directory, img_name)
            os.rename(src, dst)

        os.rmdir(augmented_subdir)
        print(f"Augmented images for {letter} saved in {output_directory}")


input_base_dir = 'Data_BSL/'
augment_images_for_alphabet(input_base_dir, num_samples=1000)
