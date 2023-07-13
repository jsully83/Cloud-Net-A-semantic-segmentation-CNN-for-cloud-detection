
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from pathlib import Path

import tensorflow as tf
import cloud_net_model
import numpy as np
import pandas as pd
import tifffile as tiff
from generators import mybatch_generator_prediction
from utils import get_input_image_names

mirrored_strategy = tf.distribute.MirroredStrategy()

def prediction():
    with mirrored_strategy.scope():
        model = cloud_net_model.model_arch(input_rows=in_rows,
                                        input_cols=in_cols,
                                        num_of_channels=num_of_channels,
                                        num_of_classes=num_of_classes)
    model.load_weights(weights_path)

    print("\nExperiment name: ", experiment_name)
    print("Prediction started... ")
    print("Input image size = ", (in_rows, in_cols))
    print("Number of input spectral bands = ", num_of_channels)
    print("Batch size = ", batch_sz)

    imgs_mask_test = model.predict_generator(
        generator=mybatch_generator_prediction(test_img, in_rows, in_cols, batch_sz, max_bit),
        steps=np.ceil(len(test_img) / batch_sz))

    print("Saving predicted cloud masks on disk... \n")

    pred_dir = experiment_name + ''
    if not os.path.exists(os.path.join(PRED_FOLDER, pred_dir)):
        os.mkdir(os.path.join(PRED_FOLDER, pred_dir))

    for image, image_id in zip(imgs_mask_test, test_ids):
        image = (image[:, :, 0]).astype(np.float32)
        tiff.imsave(os.path.join(PRED_FOLDER, pred_dir, str(image_id)), image)

# experiment_name = "MethaneSAT_Cl_0.5"
# experiment_name = 'MethaneSAT_Cl_0.99'
# experiment_name = 'MethaneSAT_Sh_0.5'
# experiment_name = 'MethaneSAT_cl01_sh05'
# experiment_name = 'MethaneSAT_cl05_sh05'
# experiment_name = 'MethaneSAT_balanced_copy'
experiment_name = 'ensemble1_aufl'

GLOBAL_PATH = str(Path().joinpath(Path().cwd().parents[0], 'data', experiment_name))
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')
PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')

in_rows = 192
in_cols = 192
num_of_channels = 4
num_of_classes = 1
batch_sz = 10
max_bit = 65535  # maximum gray level in landsat 8 images

weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.h5')


# getting input images names
test_patches_csv_name = 'test_patches_msat.csv'
df_test_img = pd.read_csv(os.path.join(TEST_FOLDER, test_patches_csv_name))
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)

prediction()
