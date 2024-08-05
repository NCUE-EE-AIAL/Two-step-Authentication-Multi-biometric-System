import os
import pandas as pd

# Parent Directory
parent_directory = '/Users/aryansmac/Documents/PROJECT/TEEP/Biometric_Authentication/Aryan_work/output_dataset'

image_ids = []
labels = []

# Walk through the parent directory
for folder_name in os.listdir(parent_directory):
    folder_path = os.path.join(parent_directory, folder_name)
    if os.path.isdir(folder_path):  # Ensure it's a directory
        for image_name in os.listdir(folder_path):
            if image_name.endswith('.jpg'):
                image_ids.append(image_name)
                labels.append(folder_name)

# Created Dataframe
dataset = pd.DataFrame({
    'image_id': image_ids,
    'label': labels
})

dataset.to_csv('output_dataset.csv', index=False)

print("done")