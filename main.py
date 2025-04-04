import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from transform_page import *

def mask_color(image, color):   

    if color == "yellow":
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define range of yellow color in HSV
        lower_yellow = np.array([10, 40, 100])
        upper_yellow = np.array([30, 255, 255])
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Bitwise-AND mask and original image
    elif color == "red":
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Define RGB range for red
        lower = np.array([0, 0, 0])     # Lower bound for deep reds
        upper = np.array([255, 60, 150]) # Upper bound for bright reds
        # Create the mask
        mask = cv2.inRange(rgb, lower, upper)

    # Apply mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def main():
    # Define the folder containing the images
    folder_path = 'Kernel_Flake_GB'
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the folder.")
        return

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        # cv2.imread(filename + '.jpg')
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not load image {filename}")
            continue

        # Display unprocessed image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        '''
        plt.imshow(img_rgb)
        plt.title(f"Unprocessed Image: {filename}")
        plt.axis('off')
        plt.show()
        '''

        try:
            warped_image = process_page(img)
        except:
            print("Could not warp image")
            continue

        # color_img = mask_color(warped_image)



if __name__ == "__main__":
    main()