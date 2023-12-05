import os
import cv2

# Specify the folder containing the images
folder_path = r'C:\Users\Andi\Research\3d-gaussian-splatting\drr\test-12-4\360-002-chest\input'

# List all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is an image
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Invert the image colors
        inverted_image = 255 - image

        # Convert to RGBA
        rgba_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2RGBA)

        # Set the alpha channel to the inverted image
        rgba_image[:, :, 3] = inverted_image

        # Overwrite the original image
        cv2.imwrite(file_path, rgba_image)
