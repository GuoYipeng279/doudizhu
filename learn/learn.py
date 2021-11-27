# import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Concatenate, Flatten, LSTM, Dense, Dropout, Bidirectional, Embedding, Input, Conv2D, Conv3D, MaxPooling2D
# from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras import Model


class Model0:
    '''神经网络模型'''
    def __init__(self, n_layers=4,
                 loss="binary_crossentropy",
                 optimizer="rmsprop",
                 metric="binary_accuracy",
                 modeling=None):  # noqa E501
        if modeling is None:
            model = Sequential()
            # model.add(Flatten())
            for i in range(n_layers):
                if i == 0:
                    # first layer
                    model.add(Dense(200, activation="relu", input_shape=(139,)))
                elif i == n_layers - 1:
                    # last layer
                    model.add(Dense(1, activation="sigmoid"))
                else:
                    # hidden layers
                    model.add(Dense(200, activation="relu"))
                # add dropout after each layer
                # model.add(Dropout(dropout))
            model.compile(loss=loss, metrics=[metric], optimizer=optimizer)
            model.summary()
            self.model = model
        else:
            self.model = load_model(modeling, custom_objects={'round_acc':metric})


class ModelC:
    '''神经网络模型'''
    def __init__(self,
                 loss="binary_crossentropy",
                 optimizer="rmsprop",
                 metric="binary_accuracy",
                 modeling=None):  # noqa E501
        if modeling is None:
            hand_input = Input(shape=(2, 15, 4, 1), name='in_hand')
            curr_input = Input(shape=(19,), name='current')
            hand_analysis1 = Conv3D(6, (2, 5, 4))(hand_input)
            hand_analysis2 = Conv3D(2, (1, 5, 1))(hand_input)
            hand_analysis3 = Conv3D(2, (2, 1, 1))(hand_input)
            # hand_analysis = Conv3D(4, (5, 4, 2))(hand_analysis)
            hand_analysis1 = Flatten()(hand_analysis1)
            hand_analysis2 = Flatten()(hand_analysis2)
            hand_analysis3 = Flatten()(hand_analysis3)
            curr_analysis = Dense(40, activation='relu')(curr_input)
            x = Concatenate(axis=-1)([hand_analysis1,
                                      hand_analysis2, hand_analysis3,
                                      curr_analysis])
            x = Dense(500, activation="relu")(x)
            x = Dense(500, activation="relu")(x)
            output = Dense(1, activation="sigmoid")(x)
            self.model = Model(inputs=[hand_input, curr_input],
                               outputs=[output])
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=[metric])
            self.model.summary()
        else:
            self.model = load_model(modeling, custom_objects={'round_acc':metric})

class ModelL:
    '''长短记忆模型'''
    def __init__(self,
                 loss="binary_crossentropy",
                 optimizer="rmsprop",
                 metric="binary_accuracy",
                 modeling=None):  # noqa E501
        if modeling is None:
            # (2, 15, 4, 1)
            hand_input = Input(shape=(15, 8), name='in_hand')
            curr_input = Input(shape=(19,), name='current')
            x = LSTM(100, return_sequences=True)(hand_input)
            x = LSTM(100)(x)
            curr_analysis = Dense(40, activation='relu')(curr_input)
            x = Concatenate(axis=-1)([x, curr_analysis])
            x = Dense(100, activation="relu")(x)
            output = Dense(1, activation="sigmoid")(x)
            self.model = Model(inputs=[hand_input, curr_input],
                               outputs=[output])
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=[metric])
            self.model.summary()
        else:
            self.model = load_model(modeling, custom_objects={'round_acc':metric})


class Binary(Callback):

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
        val_targ = self.validation_data[1].round().astype(int)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_confusion = confusion_matrix(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(epoch)
        print(_val_confusion)

        return
