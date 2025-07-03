#!/usr/bin/env python
# coding: utf-8

# PROCESSING OF HF1A IMAGES FOR EXPORT

# In[228]:


### EXPORT RGB + NIR IMAGES OF HF1A

import os
from PIL import Image
import numpy as np

# Define paths
source_dir = 'path/to/raw/hf1/images'
destination_dir = 'path/to/rgb/and/nir/images'

# Make the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Process each folder
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)

    if os.path.isdir(folder_path):
        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.png'):
                file_path = os.path.join(folder_path, file_name)
                
                # Open the image and convert it to a numpy array
                img = Image.open(file_path)
                img_array = np.array(img)
                
                # Split the image array into RGB and NIR components
                rgb_array = img_array[:, :, :3]
                nir_array = img_array[:, :, 3]
                
                # Convert arrays back to images
                rgb_image = Image.fromarray(rgb_array)
                nir_image = Image.fromarray(nir_array, 'L')  # 'L' for grayscale

                # Create new folder in destination
                new_folder_path = os.path.join(destination_dir, folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                # Save RGB and NIR images
                rgb_image.save(os.path.join(new_folder_path, f"{folder_name}_rgb.png"))
                nir_image.save(os.path.join(new_folder_path, f"{folder_name}_nir.png"))

print("Processing complete.")


# UPLOADING IMAGE FOLDERS TO ENCORD PROJECTS

# In[229]:


### UPLOAD IMAGES FROM FOLDERS AS IMAGE GROUP TO ENCORD


# Define project path and SSH key
SSH_PATH = "path/to/your/encord/key/kuva-key-private-key.txt"
PROJECT_HASH = "found-inside-project-97e3-1bd1097e2641"

# Create user client using SSH key
user_client = EncordUserClient.create_with_ssh_private_key(Path(SSH_PATH).read_text())

# Specify the Dataset you want to upload your image saequences to
dataset = user_client.get_dataset(PROJECT_HASH)

# Specify the root directory containing all your folders
root_dir = '/path/to/your/image/folders'

# Iterate through each subfolder in the root directory
for subdir in Path(root_dir).iterdir():
    if subdir.is_dir():  # Ensure it's a directory
        # Collect all image paths in the current folder
        image_paths = [str(img_path) for img_path in subdir.glob('*.png')]
        # Generate the title for the image group, ensuring it ends with ".png"
        sequence_title = f"{subdir.name}.png"  # Append '.png' to the directory name
        
        # Ensure the list is not empty
        if image_paths:
            image_paths.sort()  # Sort the image paths to maintain order
            
            # Upload the images in the current folder as one image sequence to Encord
            try:
                dataset.create_image_group(
                    image_paths,
                    create_video=False, 
                    title=sequence_title  # Using the folder name as the title
                )
                print(f"Uploaded {len(image_paths)} images from {subdir.name} as an image sequence.")
            except Exception as e:
                print(f"Failed to upload images from {subdir.name}. Error: {e}")

print("All folders have been processed and uploaded as image sequences.")


# In[6]:


# Import dependencies
from encord import EncordUserClient
from encord.objects.ontology_element import OntologyElement
from encord.objects import Object, OntologyStructure
from encord.objects.attributes import Attribute, Option
from encord.objects.coordinates import BitmaskCoordinates
from encord.objects import ObjectInstance
from collections.abc import Iterable
import subprocess
import os
from PIL import Image, ImageDraw
import requests
import io
import cv2
import numpy as np
import rasterio as rio
from rasterio.transform import from_origin, Affine
import xml.etree.ElementTree as ET


# ################################
# 
# EXPORTING CLOUD ANNOTATIONS
# 
# ################################

# In[93]:


############# EDIT THIS ONLY (PROBABLY)
base_filename = "hyperfield1a_L1B_20250519T023618.png"


# In[94]:


### Export annotations from encord to .tif

import os
import subprocess
from pathlib import Path
from osgeo import gdal
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from collections import defaultdict
from matplotlib.pyplot import imsave
from encord import EncordUserClient
from encord.client import DatasetAccessSettings
from datetime import datetime
import pandas as pd
import rasterio as rio
from affine import Affine
import cv2

##############
ssh_private_key_path = "/Users/patrickselanniemi/Desktop/Encord/kuva-key-private-key.txt"
project_hash = "6ff4fc66-2fdc-4bec-afef-fd439cc44d11"  # Edit as needed
output_folder = "/Users/patrickselanniemi/Desktop/HF1_clouds_downloaded" # Define where .pngs are downloaded
export_folder = "/Users/patrickselanniemi/Desktop/HF1_clouds_annotated"  # Define where .tifs are stored

# Location of metadata for each image
input_annotation_folder = "/Users/patrickselanniemi/Desktop/HF1_clouds_original" # Location of original .tifs for metadata

# Instantiate Encord client and access project
user_client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_private_key_path)
project = user_client.get_project(project_hash)

# Fetch and initialize labels
label_row = project.list_label_rows_v2(data_title_eq=base_filename)[0]
label_row.initialise_labels()

# Determine the directory based on the filename (without .png) in the input folder
base_dir = os.path.join(input_annotation_folder, base_filename.replace('.png', ''))

# Path to the TIFF file within the directory
tif_file_path = os.path.join(base_dir, 'L1B_rgb_nir.tif')

# Use Rasterio to open the TIFF, get dimensions, and geolocation data
with rio.open(tif_file_path) as img:
    height, width = img.height, img.width
    crs = img.crs
    transform = img.transform
    original_nodata = img.nodata  # Use a separate variable to track original nodata

# Initialize combined_mask based on the dimensions of the TIFF image
combined_mask = np.zeros((height, width), dtype=np.uint8)

# Mapping mask types to fixed grayscale intensities
intensity_map = {
    "Fill": 0,
    "Cloud_Shadow": 64,
    "Clear": 128,
    "Thin_Cloud": 192,
    "Cloud": 255
}

# Get object instances for the specific file
object_instances = label_row.get_frame_view(1).get_object_instances()
num_instances = len(object_instances)

# Loop through all instances where masks are present
for i in range(num_instances):
    object_instance = object_instances[i]
    attr = object_instance.ontology_item.attributes[0]
    mask_type = object_instance.get_answer(attr).value
    bitmask_annotation = object_instance.get_annotations()[0]
    bitmask = bitmask_annotation.coordinates.to_numpy_array().astype(np.uint8)

    # Ensure bitmask dimensions match combined_mask
    if bitmask.shape != combined_mask.shape:
        raise ValueError(f"Bitmask dimensions {bitmask.shape} do not match combined_mask dimensions {combined_mask.shape} for {base_filename}")

    # Assign intensity values based on mask type
    if mask_type in intensity_map:
        intensity = intensity_map[mask_type]
        combined_mask[bitmask == 1] = intensity

#### ADDING LOCATION METADATA TO PNG TO MAKE GEOTIF

# Create a new directory for the output, named after the image file (without extension)
output_dir_for_file = os.path.join(export_folder, base_filename.replace('.png', ''))
os.makedirs(output_dir_for_file, exist_ok=True)

# Set the output file path inside the new directory with the specific format
output_path = os.path.join(output_dir_for_file, 'L1B_annotation.tif')

# Determine the appropriate nodata value for output
# Use 255 for nodata if original_nodata is None or NaN
output_nodata = 255 if original_nodata is None or np.isnan(original_nodata) else int(original_nodata)

# Use the TIFF file to write the annotated GeoTIFF
with rio.open(output_path, "w", driver='GTiff', width=width, height=height, count=1, dtype=combined_mask.dtype, nodata=output_nodata, transform=transform, crs=crs) as dst:
    dst.write(combined_mask, 1)  # Write data to the first band

print(f"GeoTIFF created successfully at {output_path}")

# Save the combined bitmask image as a PNG
output_image_path = os.path.join(output_folder, base_filename)
cv2.imwrite(output_image_path, combined_mask)
print(f"Combined bitmask image saved to {output_image_path}")


# In[ ]:


### DATA CHECKING FOR DISCREPANCIES BETWEEN DATASETS

with rio.open("/Users/patrickselanniemi/Desktop/HF1_clouds_original/hyperfield1a_L1B_20250206T105806/L1B_rgb_nir.tif") as img:
    print(f"CRS: {img.crs}, Transform: {img.transform}, Nodata: {img.nodata}, DType: {img.dtypes}")

with rio.open("/Users/patrickselanniemi/Desktop/HF1_clouds_original/hyperfield1a_L1B_20250507T101322/L1B_rgb_nir.tif") as img:
    print(f"CRS: {img.crs}, Transform: {img.transform}, Nodata: {img.nodata}, DType: {img.dtypes}")

