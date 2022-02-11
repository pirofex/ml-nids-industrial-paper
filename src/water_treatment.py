from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

print(tf.__version__)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime

# This is an example implementation of a LSTM/CNN for the SWaT dataset from the internet.

if __name__ == '__main__':

    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    # Load the TensorBoard notebook extension
    # %load_ext tensorboard

    # Reading data from google drive
    zip_path = tf.keras.utils.get_file(
        origin='https://drive.google.com/uc?export=download&id=1klDpUNwhYp_pbUALdpKMbydBTYupIvkH',
        fname='Attack2.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)

    TRAIN_SPLIT = 8097
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    tf.random.set_seed(13)


    def create_time_steps(length):
        return list(range(-length, 0))


    def show_plot(plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                         label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel('Time-Step')
        return plt


    # Extract 3 kinds of attack
    features_considered = ['LIT 301', 'AIT 301', 'AIT 302']
    features = df[features_considered]
    features.index = df['GMT +0']
    print(features.head())

    # Draw and show the data over time for 3 attack types
    features.plot(subplots=True)

    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)

    dataset = (dataset - data_mean) / data_std


    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)


    ## Plot the training history
    def plot_train_history(history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()

        plt.show()


    past_history = 720
    future_target = 72
    STEP = 1

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)

    print('Single window of past history : {}'.format(x_train_multi[0].shape))
    print('\n Target LIT 301 value to predict : {}'.format(y_train_multi[0].shape))

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


    def multi_step_plot(history, true_future, prediction):
        plt.figure(figsize=(12, 6))
        num_in = create_time_steps(len(history))
        num_out = len(true_future)

        plt.plot(num_in, np.array(history[:, 1]), label='History')
        plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.show()


    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]))


    # Create the deep neural network model
    def create_model():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='tanh',
                                         input_shape=x_train_multi.shape[-2:]))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
        model.add(tf.keras.layers.LSTM(64))
        model.add(tf.keras.layers.Dense(units=future_target, activation='tanh'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        print(model.summary())
        return model


    multi_step_model = create_model()

    for x, y in val_data_multi.take(1):
        print(multi_step_model.predict(x).shape)

    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    EPOCHS = 5
    EVALUATION_INTERVAL = 200

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=50,
                                              callbacks=[tensorboard_callback])

    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

    multi_step_model.save('model_attack2.h5')
