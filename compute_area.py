import cv2 
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(image, mask, alpha=0.4):
    """Overlay the binary mask on top of the image with green tint."""
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 255, 0]  # green overlay for flake
    return cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)

def compute_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green background mask
    lower = np.array([30, 50, 50])
    upper = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower, upper)

    # Yellow (flake) mask
    yellow_lower = np.array([15, 40, 80])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # White (flake) mask
    white_lower = np.array([0, 0, 160])
    white_upper = np.array([180, 80, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # Final flake mask = (yellow ∪ white) ∧ ¬green
    non_green_mask = cv2.bitwise_not(green_mask)
    flake_mask = cv2.bitwise_or(yellow_mask, white_mask)
    final_mask = cv2.bitwise_and(flake_mask, non_green_mask)

    # Optional dilation (uncomment if needed)
    # kernel = np.ones((3, 3), np.uint8)
    # final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    # Show binary mask
    plt.imshow(final_mask, cmap="gray")
    plt.title("Flake Mask (binary)")
    plt.axis('off')
    plt.show()

    # Show overlay on original image
    overlay = overlay_mask(image, final_mask)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay: Flake Mask on Image")
    plt.axis('off')
    plt.show()

    object_area = np.count_nonzero(final_mask)
    total_area = image.shape[0] * image.shape[1]
    print(f"Flake pixels: {object_area}, Total: {total_area}, Ratio: {object_area / total_area:.3f}")

    return object_area / total_area * 25  # Normalize to cm² or relative scale