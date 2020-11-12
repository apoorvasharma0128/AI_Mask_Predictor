#credit :https://www.kaggle.com/andrewmvd/face-mask-detection
#credit : https://www.kaggle.com/alessiocorrado99/animals10
#credit :https://www.kaggle.com/ashwingupta3012/human-faces  ----for without mask :needs to be dowloaded

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

print(os.listdir())
print(os.getcwd())
path_image_mask=os.getcwd()+"/Database/face-mask/images/"
path_image_animals=os.getcwd()+"/Database/animals/raw-img/"
img_array = np.array(Image.open(path_image_mask+"maksssksksss0.png"))
plt.imshow(img_array)
plt.show()
