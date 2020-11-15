#credit :https://www.kaggle.com/andrewmvd/face-mask-detection
#credit : https://www.kaggle.com/alessiocorrado99/animals10
#credit :https://www.kaggle.com/ashwingupta3012/human-faces  ----for without mask :needs to be dowloaded

from PIL import Image, ImageOps
import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms

print(os.listdir())
print(os.getcwd())
image_database=os.getcwd()+"/Database"
DS_STORE='.DS_Store'
IMG_SIZE = (128,128)
RGB="RGB"
train_folder = os.getcwd()+"/Database/training-data"


def resize_image(src_image, size=IMG_SIZE, bg_color="white"):
    src_image.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new(RGB, size, bg_color)
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    return new_image

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)

for root, folders, files in os.walk(image_database):
    print(folders)
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        # Create a matching subfolder in the output dir
        saveFolder = os.path.join(train_folder,sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        file_names = os.listdir(os.path.join(root,sub_folder))
        for file_name in file_names:
            if file_name!=DS_STORE:
                file_path = os.path.join(root,sub_folder, file_name)
                image = Image.open(file_path)
                resized_image = resize_image(image, IMG_SIZE)
                saveAs = os.path.join(saveFolder, file_name)
                resized_image.save(saveAs)

def load_dataset(data_path):
    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
        # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )

    ### can be replaced by k folds
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return train_loader, test_loader
train_loader, test_loader = load_dataset(train_folder)
batch_size = train_loader.batch_size
print("Data loaders ready to read", train_folder)

