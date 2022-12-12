import pandas as pd
import numpy as np
import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow import keras
import warnings
import argparse

parser = argparse.ArgumentParser(description='classification_predict')

parser.add_argument('--dr', type=str, default="", help='Input directory')
parser.add_argument('--out_dr', type=str, default="", help='Output directory')
parser.add_argument('--type', type=str, default="", help='segmented or not')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Define functions to create a DataFrame with the filepath and the labels of the pictures
def proc_img(filepath):
    """ Process image data
    Args:
        filepath: The path of the images

    Returns: The dataframe of the path and label
    """
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepath))

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepath and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def proc_df(input_dir):
    """ Process the dataframe of the path and label
    Args:
        input_dir: The path of the images

    Returns: The predicted label
    """
    path = Path(input_dir)
    images = os.listdir(input_dir)
    filepath = [str(path) + '/' + img_path for img_path in images]
    df = proc_img(filepath)
    return df


def class_predict(dr, model_dr):
    """ classification prediction for a single model
    Args:
        dr: The path of the images
        model_dr: The path of the model

    Returns: The dataframe of the path and result
    """
    df = proc_df(dr)

    df_images_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rescale=1. / 255
    )

    df_images = df_images_generator.flow_from_dataframe(
        x_col='Filepath',
        y_col='Label',
        dataframe=df,
        target_size=(224, 224),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=False, seed=1109
    )

    model = keras.models.load_model(model_dr)

    # Evaluate the label of the test_images
    pred = model.predict(df_images, verbose=1)
    pred = np.argmax(pred, axis=1)

    return pred, df


def transfer_label(preds):
    """ Transfer number labels to names
    Args:
        preds: The predicted number labels

    Returns: The predicted label names
    """
    trans_preds = []
    for i in range(len(preds)):
        if preds[i] == 0:
            trans_preds.append("COVID19")
        elif preds[i] == 1:
            trans_preds.append("NORMAL")
        elif preds[i] == 2:
            trans_preds.append("PNEUMONIA")
        elif preds[i] == 3:
            trans_preds.append("TUBERCULOSIS")
    return trans_preds


def combine_models(dr, out_dr, type):
    """ Combine results from different models (currently 4 (1+3))
    Args:
        dr: The input path
        out_dr: The output path
        type: whether segmented or not

    Returns: The combined predicted result
    """
    # without segmentation
    # self designed CNN
    self_model_dir = "./models/self_model.h5"

    # Transfer learning
    mobilenet_model_dir = "./models/mobilenet_model.h5"
    inceptionresnetv2_model_dir = "./models/inceptionresnetv2_model.h5"
    inceptionv3_model_dir = "./models/inceptionv3_model.h5"

    model_drs = [self_model_dir,
                 mobilenet_model_dir,
                 inceptionresnetv2_model_dir,
                 inceptionv3_model_dir]

    # with segmentation
    # self designed CNN
    seg_self_model_dir = "./models/seg_self_model.h5"

    # Transfer learning
    seg_mobilenet_model_dir = "./models/seg_mobilenet_model.h5"
    seg_inceptionresnetv2_model_dir = "./models/seg_inceptionresnetv2_model.h5"
    seg_inceptionv3_model_dir = "./models/seg_inceptionv3_model.h5"

    seg_model_drs = [seg_self_model_dir,
                     seg_mobilenet_model_dir,
                     seg_inceptionresnetv2_model_dir,
                     seg_inceptionv3_model_dir]

    if type == "_seg":
        model_drs = seg_model_drs

    # predicting in each model
    preds1, df1 = class_predict(dr, model_drs[0])
    preds2, df2 = class_predict(dr, model_drs[1])
    preds3, df3 = class_predict(dr, model_drs[2])
    preds4, df4 = class_predict(dr, model_drs[3])
    pred_matrix = np.stack([preds1, preds2, preds3, preds4], axis=0)

    # combining results
    modes = []
    for i in range(pred_matrix.shape[1]):
        counts = np.bincount(pred_matrix[:, i])
        mode = np.argmax(counts)
        modes.append(mode)

    # generating labels
    preds1 = transfer_label(preds1)
    preds2 = transfer_label(preds2)
    preds3 = transfer_label(preds3)
    preds4 = transfer_label(preds4)
    modes = transfer_label(modes)
    # save results
    data = pd.DataFrame({
        "file_name": df1.iloc[:, 0],
        "self_designed_CNN": preds1,
        "mobilenet": preds2,
        "inceptionresnetv2": preds3,
        "inceptionv3": preds4,
        "combined_models": modes})
    data.to_csv(out_dr + "/predict_result" + type + ".csv", index=False)

    return modes


combine_models(args.dr, args.out_dr, args.type)
