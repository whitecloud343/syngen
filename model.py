from fastai.vision.all import *
from fastcore.all import *
from pathlib import Path
import random
import shutil
import os

# Define paths
base_path = Path(r'C:\Users\91820\Downloads\tomatofruit')  # Update this path if necessary
train_path = base_path / 'train'
valid_path = base_path / 'val'
filtered_train_path = base_path / 'filtered_train'
filtered_valid_path = base_path / 'filtered_validation'

# Ensure filtered directories exist
filtered_train_path.mkdir(parents=True, exist_ok=True)
filtered_valid_path.mkdir(parents=True, exist_ok=True)

# Function to copy images
def copy_images(src_path, dest_path, num_images=50):
    if not src_path.exists():
        raise FileNotFoundError(f"Source path {src_path} does not exist.")
    categories = src_path.ls()
    for category in categories:
        dest_category_path = dest_path / category.name
        dest_category_path.mkdir(parents=True, exist_ok=True)
        images = list(category.iterdir())
        selected_images = random.sample(images, min(num_images, len(images)))
        for img in selected_images:
            shutil.copy2(str(img), str(dest_category_path / img.name))

if __name__ == '__main__':
    # Copy images for both train and validation sets
    copy_images(train_path, filtered_train_path, num_images=50)
    copy_images(valid_path, filtered_valid_path, num_images=50)

    # Create DataLoaders
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='filtered_train', valid_name='filtered_validation'),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(base_path, bs=32)

    # Train the CNN model
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(10)

    # Save the model as a .pkl file
    model_save_path = base_path / 'tomato_fruit_model.pkl'
    learn.export(model_save_path)

    print(f"Model saved at {model_save_path}")
