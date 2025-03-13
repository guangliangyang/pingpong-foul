import os
import shutil

# Set the source directory where your images and text files are storedv
source_directory = 'C:\\workspace\\datasets\\video_extracted'

# Set the destination directories for images and labels
image_dest = "C:\\workspace\\datasets\\coco-pp\\images\\train"
label_dest = "C:\\workspace\\datasets\\coco-pp\\labels\\train"

# Ensure destination directories exist
os.makedirs(image_dest, exist_ok=True)
os.makedirs(label_dest, exist_ok=True)

# List all files in the source directory
files = os.listdir(source_directory)

# Create a set of all txt file names (without extension)
txt_files = {os.path.splitext(file)[0] for file in files if file.endswith('.txt')}

# Loop through all jpg files and copy to the destination if a matching txt file exists
for file in files:
    if file.endswith('.jpg'):
        # Get the file name without extension
        file_name = os.path.splitext(file)[0]
        # Check if a corresponding txt file exists
        if file_name in txt_files:
            # Copy jpg file to image destination
            shutil.copy(os.path.join(source_directory, file), image_dest)
            print(f"Copied image: {file}")

            # Copy corresponding txt file to label destination
            txt_file = f"{file_name}.txt"
            shutil.copy(os.path.join(source_directory, txt_file), label_dest)
            print(f"Copied label: {txt_file}")
