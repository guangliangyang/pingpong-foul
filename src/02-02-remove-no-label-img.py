import os

# Set the directory where your images and text files are stored
directory = 'C:\\workspace\\datasets\\video_extracted'

# List all files in the directory
files = os.listdir(directory)

# Create a set of all txt file names (without extension)
txt_files = {os.path.splitext(file)[0] for file in files if file.endswith('.txt')}

# Loop through all jpg files
for file in files:
    if file.endswith('.jpg'):
        # Get the file name without extension
        file_name = os.path.splitext(file)[0]
        # If there's no corresponding txt file, remove the jpg file
        if file_name not in txt_files:
            os.remove(os.path.join(directory, file))
            print(f"Removed: {file}")
