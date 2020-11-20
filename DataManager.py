#credit :https://www.kaggle.com/andrewmvd/face-mask-detection
#credit : https://www.kaggle.com/alessiocorrado99/animals10
#credit :https://www.kaggle.com/ashwingupta3012/human-faces  ----for without mask :needs to be dowloaded

from PIL import Image, ImageOps
import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold


DS_STORE='.DS_Store'
RGB="RGB"
IMG_SIZE = (128,128)
train_folder = os.getcwd()+"/Database/training-data"
image_database=os.getcwd()+"/Database"

class DataManager:
    def resizeImage(src_image, size, bg_color="white"):
        try:
            src_image.thumbnail(size, Image.ANTIALIAS)
            new_image = Image.new(RGB, size, bg_color)
            new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
            return new_image
        except:
            print("Issue occured while resizing images")

    def transformImages(self):
        try:
            print(os.path.exists(train_folder))
            if os.path.exists(train_folder):
                shutil.rmtree(train_folder)
            for root, folders, files in os.walk(image_database):
                print("root"+root)
                print("Classes :",folders)
                for sub in folders:
                    print('processing folder ' + sub)
                    newLoc = os.path.join(train_folder,sub)
                    if not os.path.exists(newLoc):
                        os.makedirs(newLoc)
                    file_names = os.listdir(os.path.join(root,sub))
                    for file in file_names:
                        if file!=DS_STORE:
                            resized_image = self.resizeImage(Image.open(os.path.join(root,sub,file)),IMG_SIZE)
                            newPath = os.path.join(newLoc, file)
                            resized_image.save(newPath)

            print("Transformation Step complete")
        except:
            print("Error Occured in TransformData")


    def loadImages(data_path):
        transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transformation)
        train_size = int(0.6 * len(dataset))
        test_size = len(dataset) - train_size
        # use torch.utils.data.random_split for training/test split
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=50,num_workers=2,shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=50,num_workers=2,shuffle=False)
        return train_loader, test_loader
