
from __future__ import print_function

import os


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


from pathlib import Path

import cloud_net_model
import numpy as np
import pandas as pd
import tensorflow as tf

from generators import mybatch_generator_train, mybatch_generator_validation
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from losses import jacc_coef, AsymmetricUnifiedFocalLoss

from sklearn.model_selection import train_test_split
from utils import ADAMLearningRateTracker, get_input_image_names

mirrored_strategy = tf.distribute.MirroredStrategy()

def train():
    with mirrored_strategy.scope():
        metrics = [jacc_coef, 
                  tf.keras.metrics.BinaryIoU(name='IoU'),
                  tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                  tf.keras.metrics.AUC(name='prc', curve='PR'),
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall')]
                #   Add metrics for F1Score and TP, FP, FN, TN for the confusion matrix

        asym_uni_focal = AsymmetricUnifiedFocalLoss(loss_weight, loss_delta, loss_gamma)

        model = cloud_net_model.model_arch(input_rows=in_rows,      
                                        input_cols=in_cols,
                                        num_of_channels=num_of_channels,
                                        num_of_classes=num_of_classes)
        model.compile(optimizer=Adam(learning_rate=starting_learning_rate), loss=asym_uni_focal, metrics=metrics)
    # model.summary()

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger(experiment_name + '_log_1.log')

    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)

    if train_resume:
        model.load_weights(weights_path)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")

    print("Experiment name: ", experiment_name)
    print("Input image size: ", (in_rows, in_cols))
    print("Number of input spectral bands: ", num_of_channels)
    print("Learning rate: ", starting_learning_rate)
    print("Batch size: ", batch_sz, "\n")
    print(tf.config.list_physical_devices('GPU'))

    model.fit_generator(
        generator=mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_sz), epochs=max_num_epochs, verbose=1,
        validation_data=mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        validation_steps=np.ceil(len(val_img_split) / batch_sz),
        callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate), csv_logger])

experiment_name = "ensemble1_aufl"
GLOBAL_PATH = str(Path().joinpath(Path().cwd().parents[0], 'data', experiment_name))
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')

in_rows = 192
in_cols = 192
num_of_channels = 4
num_of_classes = 1
starting_learning_rate = 1e-4
# starting_learning_rate = 0.7e-4
end_learning_rate = 1e-8
max_num_epochs = 2000  # just a huge number. The actual training should not be limited by this value
val_ratio = 0.2
patience = 15
decay_factor = 0.7
# batch_sz = 12
batch_sz = 8
max_bit = 65535  # maximum gray level in landsat 8 images
loss_weight=0.5
loss_delta=0.6
loss_gamma=0.5

weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.h5')
train_resume = False

# getting input images names
train_patches_csv_name = 'training_patches_msat.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

train()
