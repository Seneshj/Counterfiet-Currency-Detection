import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras.preprocessing.image

# Define the data generator
from keras.utils import load_img, img_to_array, array_to_img

data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load the images
main_folder = "C:\\Users\\yohan\\OneDrive\\Desktop\\DataSet"
notes_folder = os.path.join(main_folder, "Notes")
coins_folder = os.path.join(main_folder, "Coins")

# Define the folder where the augmented images will be saved
new_folder = "C:\\Users\\yohan\\OneDrive\\Desktop\\NewDataset"
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# Iterate through the sub folders inside the "Notes" folder
for sub_folder in os.listdir(notes_folder):
    print(f"Entered {sub_folder}")
    sub_folder_path = os.path.join(notes_folder, sub_folder)
    sub_folder_new_path = os.path.join(new_folder, "Notes", sub_folder)
    if not os.path.exists(sub_folder_new_path):
        os.makedirs(sub_folder_new_path)
    for img_file in os.listdir(sub_folder_path):
        print(f"Now augmenting {img_file}")
        img = load_img(os.path.join(sub_folder_path, img_file))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_gen.flow(x, batch_size=1):
            i += 1
            img = array_to_img(batch[0])
            img.save(os.path.join(sub_folder_new_path, "image_aug_" + str(i) + "_" + img_file))
            if i >= 4:
                break

# Iterate through the sub folders inside the "Notes" folder
for sub_folder in os.listdir(coins_folder):
    print(f"Entered {sub_folder}")
    sub_folder_path = os.path.join(coins_folder, sub_folder)
    sub_folder_new_path = os.path.join(new_folder, "Coins", sub_folder)
    if not os.path.exists(sub_folder_new_path):
        os.makedirs(sub_folder_new_path)
    for img_file in os.listdir(sub_folder_path):
        print(f"Now augmenting {img_file}")
        img = load_img(os.path.join(sub_folder_path, img_file))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_gen.flow(x, batch_size=1):
            i += 1
            img = array_to_img(batch[0])
            img.save(os.path.join(sub_folder_new_path, "image_aug_" + str(i) + "_" + img_file))
            if i >= 4:
                break
