import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional

CuDNNLSTM = tf.compat.v1.keras.layers.CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from collections import Counter
from PIL import Image
from itertools import groupby

image_paths = []
image_texts = []

data_folder = "../spark-ai-summit-2020-text-extraction/mjsynth_sample"

for path in os.listdir(data_folder):
    image_paths.append(data_folder + "/" + path)
    image_texts.append(path.split("_")[1])

incorrupt_images = []

for path in image_paths:
    try:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    except:
        incorrupt_images.append(path)

for path in incorrupt_images:
    incorrupt_index = image_paths.index(path)
    del image_paths[incorrupt_index]
    del image_texts[incorrupt_index]

vocab = set("".join(map(str, image_texts)))

max_label_len = max([len(str(text)) for text in image_texts])

char_list = sorted(vocab)


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []

    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return pad_sequences([dig_lst], maxlen=max_label_len, padding='post', value=len(char_list))[0]


padded_image_texts = list(map(encode_to_labels, image_texts))

train_image_paths = image_paths[: int(len(image_paths) * 0.90)]
train_image_texts = padded_image_texts[: int(len(image_texts) * 0.90)]

val_image_paths = image_paths[int(len(image_paths) * 0.90):]
val_image_texts = padded_image_texts[int(len(image_texts) * 0.90):]


def process_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)

    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)

    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 4. Resize to the desired size
    img = tf.image.resize(img, [32, 128])

    return {"image": img, "label": label}


batch_size = 256

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_texts))

train_dataset = (
    train_dataset.map(
        process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_image_texts))
validation_dataset = (
    validation_dataset.map(
        process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_list, num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True, num_oov_indices=0
)


class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time losses value and add it
        # to the layer using `self.add_loss()`.

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def ctc_decoder(predictions):
    '''
    input: given batch of predictions from text rec model
    output: return lists of raw extracted text

    '''
    text_list = []

    pred_indcies = np.argmax(predictions, axis=2)

    for i in range(pred_indcies.shape[0]):
        ans = ""

        ## merge repeats
        merged_list = [k for k, _ in groupby(pred_indcies[i])]

        ## remove blanks
        for p in merged_list:
            if p != len(char_list):
                ans += char_list[int(p)]

        text_list.append(ans)

    return text_list


figures_list = []


class PlotPredictions(tf.keras.callbacks.Callback):

    def __init__(self, frequency=1):
        self.frequency = frequency
        super(PlotPredictions, self).__init__()

        batch = validation_dataset.take(1)
        self.batch_images = list(batch.as_numpy_iterator())[0]["image"]
        self.batch_labels = list(batch.as_numpy_iterator())[0]["label"]

    def plot_predictions(self, epoch):

        prediction_model = keras.models.Model(
            self.model.get_layer(name="image").input,
            self.model.get_layer(name="dense").output
        )

        preds = prediction_model.predict(self.batch_images)
        pred_texts = ctc_decoder(preds)

        orig_texts = []

        for label in self.batch_labels:
            orig_texts.append(
                "".join([char_list[int(char_ind)] for char_ind in label if not (char_ind == len(char_list))]))

        fig, ax = plt.subplots(4, 4, figsize=(15, 5))
        fig.suptitle('Epoch: ' + str(epoch), weight='bold', size=14)

        for i in range(16):
            img = (self.batch_images[i, :, :, 0] * 255).astype(np.uint8)
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

        plt.show()
        # plt.savefig("predictions_epoch_"+ str(epoch)+'.png', bbox_inches = 'tight', pad_inches = 0)

        figures_list.append(fig)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            self.plot_predictions(epoch)


def train(epochs):
    # input with shape of height=32 and width=128
    inputs = Input(shape=(32, 128, 1), name="image")

    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    conv_1 = Conv2D(32, (3, 3), activation="selu", padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)  # 16 64

    conv_2 = Conv2D(64, (3, 3), activation="selu", padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)  # 8 32

    conv_3 = Conv2D(128, (3, 3), activation="selu", padding='same')(pool_2)
    conv_4 = Conv2D(128, (3, 3), activation="selu", padding='same')(conv_3)

    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)  # 4 32

    conv_5 = Conv2D(256, (3, 3), activation="selu", padding='same')(pool_4)

    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(256, (3, 3), activation="selu", padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)  # 2 32

    conv_7 = Conv2D(64, (2, 2), activation="selu")(pool_6)  # 1 31

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)  # 31,512

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
    blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)

    softmax_output = Dense(len(char_list) + 1, activation='softmax', name="dense")(blstm_2)

    output = CTCLayer(name="ctc_loss")(labels, softmax_output)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

    # model to be used at training time
    model = Model(inputs=[inputs, labels], outputs=output)
    model.compile(optimizer=optimizer)

    print(model.summary())
    file_path = "C_LSTM_best.hdf5"

    checkpoint = ModelCheckpoint(filepath=file_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    callbacks_list = [checkpoint,
                      # WandbCallback(monitor="val_loss",
                      #               mode="min",
                      #               log_weights=True),
                      PlotPredictions(frequency=1),
                      EarlyStopping(patience=3, verbose=1)]

    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        verbose=1,
                        callbacks=callbacks_list,
                        shuffle=True)

    return model


model = train(epochs=30)

model.load_weights('C_LSTM_best.hdf5')

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense").output
)
prediction_model.summary()
