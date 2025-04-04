import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_black_areas(image):
    """Creates a black mask to isolate black elements (corner markers) and remove background."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 134])  # Adjust if needed

    # Create the black mask
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # Apply Morphological Opening (Removes small noise)
    kernel = np.ones((3, 3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    # Apply Morphological Closing (Fills small gaps)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    # Show the refined black mask
    plt.imshow(black_mask, cmap="gray")
    plt.title("Refined Black Mask")
    plt.axis('off')
    plt.show()

    return black_mask

def find_corner_coords(image):
    """Finds the four black corner markers using the black mask."""
    black_mask = filter_black_areas(image)

    # Apply slight blurring to reduce noise in contour detection
    blurred_mask = cv2.GaussianBlur(black_mask, (5, 5), 0)

    # Find contours in the black mask
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define the expected characteristics of the markers
    min_area = 1000  # Minimum contour area for a valid marker
    max_area_ratio = 0.2  # Max area relative to the full image
    w, h, _ = image.shape
    max_area = w * h * max_area_ratio

    detected_markers = []

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 6:  # The page corner markers have 6 vertices
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                detected_markers.append((approx, area))

    # Sort by area and find the best four markers
    detected_markers = sorted(detected_markers, key=lambda x: x[1])
    
    if len(detected_markers) < 4:
        print("Not enough markers detected.")
        return None

    best_four = detected_markers[:4]

    # Extract centroid coordinates of the four detected markers
    corner_coords = np.empty((0, 2), dtype="float32")
    for contour, _ in best_four:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        corner_coords = np.vstack([corner_coords, [cx, cy]])

        # Draw detected markers on the image
        cv2.circle(image, (cx, cy), 10, (255, 0, 0), -1)

    # Show detected corner coordinates
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Detected Corner Markers")
    plt.axis('off')
    plt.show()

    return corner_coords

def transform_page(image, corners):
    """Applies perspective transformation using detected corners."""
    output_width, output_height = 390, 440  # Dimensions for white page (19.5 cm × 22 cm)

    # Define the four destination points for warping
    pts_dst = np.array([
        [0, 0], 
        [output_width - 1, 0], 
        [0, output_height - 1], 
        [output_width - 1, output_height - 1]
    ], dtype="float32")

    # Ensure corners are correctly matched to their locations
    sorted_corners = np.empty((0, 2), dtype="float32")

    for point_src in pts_dst:
        distances = np.linalg.norm(corners - point_src, axis=1)
        closest_index = np.argmin(distances)
        sorted_corners = np.vstack([sorted_corners, corners[closest_index]])
        corners = np.delete(corners, closest_index, 0)

    # Compute transformation matrix
    try:
        M = cv2.getPerspectiveTransform(np.float32(sorted_corners), np.float32(pts_dst))
    except:
        print("getpersective transform")

    # Apply the transformation
    warped_image = cv2.warpPerspective(image, M, (output_width, output_height))

    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped Page")
    plt.axis('off')
    plt.show()

    return warped_image

def process_page(image):
    """Pipeline for detecting and transforming a white page using black markers."""
    corners = find_corner_coords(image)
    print(corners)
    if corners is None:
        return None

    warped = transform_page(image, corners)
    return warped
