""""functions for spot classification

Notes: assumes data structure (spots, z, y, x), with currently (z,y,x) = (9,11,11)

TODO: include spot prediciton prob in spots file
"""

import numpy as np
from scipy import ndimage as ndi
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import random
import glob
import pickle
from sklearn.model_selection import train_test_split, KFold
import pandas as pd


def normalize_data(data):
    """ normalize voxel intensities to (0,1)"""
    data_norm = np.zeros_like(data)
    for spot_ind in range(len(data)):
        data_norm[spot_ind] = (data[spot_ind] - np.min(data[spot_ind])) / (
            np.max(data[spot_ind] - np.min(data[spot_ind])))

    return data_norm


def make_cnn(width=11, height=11, depth=9, n_filters1=4, n_filters2=4):
    """Build a 3D convolutional neural network model. """
    # keep it zyx
    inputs = keras.Input((depth, width, height, 1))

    x = layers.Conv3D(filters=n_filters1, kernel_size=(3, 3, 3), activation="relu", padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=n_filters2, kernel_size=(3, 3, 3), activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")

    return model


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees. decorator is used by tensorflow to run during data loading."""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndi.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def make_prediction(model, data):
    """use a trained model to classify spot or not spot"""
    data_norm = normalize_data(data)

    results = model.predict(data_norm)

    return results


def run_batch_prediction(df, path_to_model):
    """run classifier on every spot in a dataframe. model=keras model."""
    #data, labels = create_training_data_from_spots_df(df)
    data = np.array(df.data.to_list()).astype('float32')

    # load model
    model = keras.models.load_model(path_to_model)

    # pre process data
    data_norm = normalize_data(data)
    #data_norm = np.reshape(data_norm,
    #                       (data_norm.shape[0], data_norm.shape[2], data_norm.shape[3], data_norm.shape[1]))
    X = np.array(data_norm, dtype='float32')
    X = tf.expand_dims(X, axis=4)

    # predict. results = probability of being a true spot.
    results = model.predict(X, batch_size=32)

    # update spots
    df['prob'] = results

    return df


def run_batch_prediction_by_time_point(df, path_to_model):
    """run classifier on every spot in a dataframe but do it in chunks, by time point. model=keras model."""
    time_points = np.unique(df.t)
    prob_df = pd.DataFrame(np.zeros(len(df)), columns=['prob'], dtype='float32', index=df.index)
    for t in time_points:
        print(f'running prediction for t = {t}')
        this_df = df[df.t == t]
        data, labels = create_training_data_from_spots_df(this_df)
        data = data.astype('float32')

        # load model
        model = keras.models.load_model(path_to_model)

        # pre process data
        data_norm = normalize_data(data)
        # data_norm = np.reshape(data_norm,
        #                       (data_norm.shape[0], data_norm.shape[2], data_norm.shape[3], data_norm.shape[1]))
        X = np.array(data_norm, dtype='float32')
        X = tf.expand_dims(X, axis=4)

        # predict. results = probability of being a true spot.
        results = model.predict(X, batch_size=32)

        # update spots
        prob_df.loc[this_df.index, 'prob'] = results.flatten()

    df = pd.concat((df, prob_df), axis=1)

    return df


def create_training_data_from_spots_df(df, save_dir=None, manual_labels=False):
    """extract pixel data from spots dataframe into a 4D array (spot,z,y,x) and also extract labels into a list.
    This function is used in two contexts: (1) creating data and label arrays for training and (2) create just data
    arrays for batch prediction. In the first case, pass manual_labels=True, and this will extract only the data and
    labels for which None has been manually changed to True or False. In this case, also pass a save_dir to save the
    data for training. In the second case, pass manual_labels=False, and it will extract data for all spots."""

    # check to see if any spots have been classified
    if not any(df.manual_classification) and manual_labels:
        raise ValueError('no spots have been manually classified. aborting.')

    data = np.array(df.data.to_list())
    labels = df.manual_classification.to_list()

    if manual_labels:
        data = data[labels is not None]
        labels = labels[labels is not None]

    if save_dir is not None and manual_labels:
        with open(save_dir + '/training_data.pkl', "wb") as data_file:
            pickle.dump(data, data_file)

        with open(save_dir + '/training_labels.pkl', "wb") as labels_file:
            pickle.dump(labels, labels_file)

    return data, labels


def train_model(df, model, learning_rate=1e-4, batch_size=8, epochs=100, test_size=0.33):
    # remove spots that were not classified
    df = df[[label is not None for label in df.manual_classification]]

    # load data and labels into arrays
    data = np.array(df.data.to_list())
    labels = np.array(df.manual_classification.to_list(), dtype=int)
    data_norm = normalize_data(data)

    # Compile model.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    # assemble data into proper arrays for the model
    #data_norm = np.reshape(data_norm, (data_norm.shape[0], data_norm.shape[2], data_norm.shape[3], data_norm.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(data_norm, labels, test_size=test_size)
    print(f'number of true spots in the test set is {np.sum(y_test)}')
    X_train = np.array(X_train, dtype='float32')
    X_test = np.array(X_test, dtype='float32')

    # code from ct scan example #
    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Augment on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(X_train))
            .map(train_preprocessing)
            .batch(batch_size)
            .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(X_test))
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(2)
    )

    # class weights --- helps a lot with the class imbalance problem!
    neg, pos = np.bincount(labels)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # fit
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, class_weight=class_weight)
    #history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

    return history


def train_model_kfold(df, models_dir, n_splits=5, n_filters1=4, n_filters2=4, learning_rate=1e-4, batch_size=8,
                      epochs=100):
    # load the data and labels into arrays
    data = np.array(df.data.to_list()).astype('float32')
    labels = np.array(df.manual_classification.to_list(), dtype=int)

    # normalize the data
    data_norm = normalize_data(data)

    # define the KFold
    kf = KFold(n_splits=n_splits)

    # define a counter to keep track of which fold we are one
    fold = 1

    # loop over splits
    for train_index, val_index in kf.split(np.zeros(len(df)), labels):
        # make the model
        model = make_cnn(n_filters1=n_filters1, n_filters2=n_filters2)

        # Compile model.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy",
                      metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

        # load the data
        X_train = data_norm[train_index].astype('float32')
        X_test = data_norm[val_index].astype('float32')
        y_train = labels[train_index]
        y_test = labels[val_index]

        # code from ct scan example #
        # Define data loaders.
        train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Augment on the fly during training.
        train_dataset = (
            train_loader.shuffle(len(X_train))
                .map(train_preprocessing)
                .batch(batch_size)
                .prefetch(2)
        )
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(X_test))
                .map(validation_preprocessing)
                .batch(batch_size)
                .prefetch(2)
        )

        # class weights --- helps a lot with the class imbalance problem!
        neg, pos = np.bincount(labels)
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(models_dir + '/' + get_model_name(fold),
                                                        monitor='val_accuracy', verbose=1,
                                                        save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # fit
        history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, class_weight=class_weight,
                            callbacks=callbacks_list)

        # save history as well
        with open(models_dir + f'/history_{fold}.pkl', 'wb') as file:
            pickle.dump(history.history, file)

        fold += 1

    return


def get_model_name(fold):
    return f'model_{fold}.h5'
