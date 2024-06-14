import os
import pandas as pd

# Define the path to the root directory containing subfolders of skin conditions
root_dir = 'images'

# Initialize a list to store image paths and their corresponding conditions
data = []

# Traverse the directory structure
for condition in os.listdir(root_dir):
    condition_path = os.path.join(root_dir, condition)
    if os.path.isdir(condition_path):
        for image_name in os.listdir(condition_path):
            image_path = os.path.join(condition_path, image_name)
            if os.path.isfile(image_path):
                data.append([image_path, condition])

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=['image_path', 'condition'])

# Save the DataFrame to a CSV file
output_csv = 'skin_conditions_labels.csv'
df.to_csv(output_csv, index=False)

print(f'CSV file "{output_csv}" created successfully.')
