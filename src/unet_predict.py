import os
import time
from skimage.io import imsave
from tensorflow import keras
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import argparse

parser = argparse.ArgumentParser(description='unet_predict')

parser.add_argument('--dr', type=str, default="", help='Input directory')
parser.add_argument('--out_dr', type=str, default="", help='Output directory')
parser.add_argument('--unet_dr', type=str, default="", help='Unet model directory')

args = parser.parse_args()

warnings.filterwarnings('ignore')

#define the prediction function
def prediction(dr,out_dr,unet_dr):
    """The main prediction function

    Args:
        dr: The path of input folder
        out_dr: The path of output folder
        unet_dr: unet directory

    Returns: prediction result
    """
    # start algorithm
    print("Algorithm starting...")

    # extracting files
    images = os.listdir(dr)

    # Sort images and masks
    images.sort()

    # obtain original image, resized image, and image original size
    print("Fetching figures...")
    x1 = []
    x2 = []
    x3 = []
    for i in tqdm(range(len(images))):
        print("Fetching loop:" + str(i))
        image = images[i]
        img = Image.open(os.path.join(dr,image)).convert('L') # Convert to grayscale
        # original image
        x1.append(np.asarray(img))
        # resized image
        img_resize = np.asarray(img.resize((128,128)))/255. # Normalization
        x2.append(img_resize)
        # image original size
        img_tmp = np.asarray(img)
        size = img_tmp.shape
        x3.append(np.asarray(size))
        time.sleep(0.01)

    print("Complete looping...")

    # Transfer into array
    orig_img = np.array(x1)
    resized_images = np.array(x2)
    orig_sizes = np.array(x3)

    print("Array obtained...")

    # Predict
    print("Start predicting...")
    u_net= keras.models.load_model(unet_dr)
    for i in tqdm(range(len(resized_images))):
        print("Predicting loop:" + str(i))
        img = resized_images[i]
        # Expand the shape of array to satisfy the input requirement of the model
        mask = u_net.predict(np.expand_dims(img,axis=0))
        # resize
        mask_resize = cv2.resize(np.asarray(np.squeeze(mask)), np.flip(orig_sizes[i]))
        # squeeze
        segmented = np.squeeze(orig_img[i]).copy()
        segmented[np.squeeze(mask_resize)<0.2] = 0
        imsave(out_dr + '/' + '{n}_segmented.jpg'.format(n=images[i].split('.')[0]),segmented)
        time.sleep(0.01)

prediction(args.dr, args.out_dr, args.unet_dr)
