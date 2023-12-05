import os
import pydicom
import numpy as np

def convert_dicom_to_npz(dicom_folder, npz_filename):
    """
    Convert DICOM images in a folder to a single NPZ file.

    Args:
    dicom_folder (str): Path to the folder containing DICOM files.
    npz_filename (str): Path to the output NPZ file.
    """
    images = []
    for filename in sorted(os.listdir(dicom_folder)):
        if filename.endswith('.dcm'):
            filepath = os.path.join(dicom_folder, filename)
            dicom_image = pydicom.dcmread(filepath)
            pixel_array = dicom_image.pixel_array
            images.append(pixel_array)

    images_np = np.array(images)
    np.savez_compressed(npz_filename, data=images_np)

# Example usage
dicom_folder = r'C:\Users\Andi\Research\3d-gaussian-splatting\data\manifest-1600709154662\LIDC-IDRI\LIDC-IDRI-0002\01-01-2000-NA-NA-98329\3000522.000000-NA-04919'  # Replace with your DICOM folder path
npz_filename = r'C:\Users\Andi\Research\3d-gaussian-splatting\drr\test-12-4\npz_data\output002.npz'       # Replace with your desired output file name
convert_dicom_to_npz(dicom_folder, npz_filename)
