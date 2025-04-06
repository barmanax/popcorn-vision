import cv2
import matplotlib.pyplot as plt

def segment_3x3_grid(image):
    h, w, _ = image.shape
    
    # Crop to remove margins (adjusted empirically)
    image = image[52:362, 40:w-40]

    # Show cropped image
    '''plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Warped Image")
    plt.axis('off')
    plt.show()'''
    
    cell_height = image.shape[0] // 3
    cell_width = image.shape[1] // 3
    selected_indices = [0, 2, 4, 6, 8]
    cells = []

    for idx in selected_indices:
        row = idx // 3
        col = idx % 3
        y1, y2 = row * cell_height, (row + 1) * cell_height
        x1, x2 = col * cell_width, (col + 1) * cell_width
        cell = image[y1:y2, x1:x2]
        cells.append((idx, cell))

    # Debug: Show selected cells
    '''for idx, cell_img in cells:
        plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Cell Index: {idx}")
        plt.axis('off')
        plt.show()'''

    return cells
