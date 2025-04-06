import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from transform_page import warp_image_to_bounding_box
from segment_cells import segment_3x3_grid
from compute_area import compute_area

def main():
    folder_path = 'Kernel_Flake_GB'

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Sort to ensure kernel_n matches flake_n
    kernel_images = sorted([f for f in os.listdir(folder_path) if f.startswith("Kernel")])
    flake_images = sorted([f for f in os.listdir(folder_path) if f.startswith("Flake")])

    data = []
    labels = ['A', 'B', 'C', 'D', 'E']  # Corresponding to indices 0, 2, 4, 6, 8

    for kernel_file, flake_file in zip(kernel_images, flake_images):
        image_id = kernel_file.split('_')[-1].split('.')[0]  # e.g., '1' from 'Kernel_1.jpg'
        kernel_img = cv2.imread(os.path.join(folder_path, kernel_file))
        flake_img = cv2.imread(os.path.join(folder_path, flake_file))

        if kernel_img is None or flake_img is None:
            print(f"Could not load pair: {kernel_file}, {flake_file}")
            continue

        # Warp both images
        warped_kernel = warp_image_to_bounding_box(kernel_img)
        warped_flake = warp_image_to_bounding_box(flake_img)

        if warped_kernel is None or warped_flake is None:
            print(f"Could not warp pair: {kernel_file}, {flake_file}")
            continue

        # Segment both into 3x3 grids
        kernel_cells = segment_3x3_grid(warped_kernel)
        flake_cells = segment_3x3_grid(warped_flake)

        # Loop through the 5 positions: (0,2,4,6,8)
        for (cell_idx_k, kernel_cell), (cell_idx_f, flake_cell), label in zip(kernel_cells, flake_cells, labels):
            kernel_area = compute_area(kernel_cell)
            flake_area = compute_area(flake_cell)

            data.append({
                "image_id": int(image_id),
                "cell_id": cell_idx_k,  # same as cell_idx_f
                "label": label,
                "kernel_area": kernel_area,
                "flake_area": flake_area
            })

    df = pd.DataFrame(data)
    df.sort_values(by=["image_id", "cell_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save to CSV
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/kernel_flake_areas.csv", index=False)
    print("Saved kernel_flake_areas.csv!")

if __name__ == "__main__":
    main()