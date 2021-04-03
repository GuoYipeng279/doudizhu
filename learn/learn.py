# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, LSTM, Dense, Dropout, Bidirectional, Embedding, Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D
# from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class Model:
    '''神经网络模型'''
    def __init__(self, n_layers=3, loss="binary_crossentropy", optimizer="rmsprop"):  # noqa E501
        model = Sequential()
        # model.add(Flatten())
        for i in range(n_layers):
            if i == 0:
                # first layer
                model.add(Dense(200, activation="relu", input_shape=(200,)))
            elif i == n_layers - 1:
                # last layer
                model.add(Dense(200, activation="relu"))
            else:
                # hidden layers
                model.add(Dense(200, activation="relu"))
            # add dropout after each layer
            # model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss=loss, metrics=["binary_accuracy"], optimizer=optimizer)
        model.summary()
        self.model = model


class Metrics(Callback):

    def __init__(self, val_data, batch_size=20):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round().astype(int)
        val_targ = self.validation_data[1].astype(int)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_confusion = confusion_matrix(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(_val_confusion)

        return
