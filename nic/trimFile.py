import pandas as pd

# Load the CSV file
file_path = "NF-CSE-CIC-IDS2018-v2-ori.csv"
data = pd.read_csv(file_path)

# Keep only the first 200000 rows
trimmed_data = data.head(2000000)

# Save the trimmed data to a new CSV file
trimmed_file_path = "NF-CSE-CIC-IDS2018-v2_200w.csv"
trimmed_data.to_csv(trimmed_file_path, index=False)

