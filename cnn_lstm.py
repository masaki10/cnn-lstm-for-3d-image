from random import shuffle
from sqlite3 import Time
import efficientnet_3D.tfkeras as efn
from keras.models import Sequential, load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import MaxPooling3D, Conv3D, MaxPooling2D, Conv2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ConvLSTM2D, ConvLSTM3D
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger, ModelCheckpoint
# from keras.utils import Sequence
from sklearn.model_selection import train_test_split
import pickle
import os
import csv
import tensorflow as tf
import tensorflow_hub as hub
from generator import Generator
import config as cf
from cnn_models.vgg16 import vgg16
from cnn_models.vgg19 import vgg19
from cnn_models.resnet import Resnet3DBuilder

class CNNLSTM:
    def __init__(self):
        self.input_shape = cf.INPUT_SHAPE
        self.is_pretrained = cf.IS_PRETRAINED
        self._build_model()
        self.model.summary()
        self.train_generator = Generator("train")
        self.val_generator = Generator("val")
        self.test_generator = Generator("test")

    def _get_efficientnet(self):
        if self.is_pretrained:
            efn_model = efn.EfficientNetB0(input_shape=self.input_shape, weights='imagenet')
        else:
            efn_model = efn.EfficientNetB0(input_shape=self.input_shape, weights=None)
        return efn_model

    def _get_vgg16(self):
        return vgg16(cf.INPUT_SHAPE)

    def _get_vgg19(self):
        return vgg19(cf.INPUT_SHAPE)

    def _get_resnet50(self):
        return Resnet3DBuilder.build_resnet_50(cf.INPUT_SHAPE)

    def _get_resnet101(self):
        return Resnet3DBuilder.build_resnet_101(cf.INPUT_SHAPE)

    def _build_model(self):
        if cf.CNN_MODEL == "EfficientNet":
            cnn_model = self._get_efficientnet()
        elif cf.CNN_MODEL == "VGG16":
            cnn_model = self._get_vgg16()
        elif cf.CNN_MODEL == "VGG19":
            cnn_model = self._get_vgg19()
        elif cf.CNN_MODEL == "ResNet50":
            cnn_model = self._get_resnet50()
        elif cf.CNN_MODEL == "ResNet101":
            cnn_model = self._get_resnet101()
        else:
            raise

        cnn_output_shape = cnn_model.output_shape
        print("============================")
        print(cnn_output_shape)
        print("============================")
        cnn_model.compute_output_shape = lambda x : (x[0], cnn_output_shape[1], cnn_output_shape[2],
                                                        cnn_output_shape[3], cnn_output_shape[4])

        self.model = Sequential()
        self.model.add(TimeDistributed(cnn_model,
                input_shape=(cf.FRAME, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])))
        # self.model.add(TimeDistributed(GlobalAveragePooling3D()))
        # self.model.add(LSTM(128, return_sequences=False))
        self.model.add(ConvLSTM3D(cf.CONV_LSTM_FILTER_NUM, cf.CONV_LSTM_KERNEL_SIZE, padding="same"))
        self.model.add(GlobalAveragePooling3D())
        self.model.add(Dense(3))
        self.model.compile(optimizer=Adam(learning_rate=cf.LEARNING_RATE),
                            loss=cf.LOSS_FUNCTION, metrics=["mean_absolute_error"])
        
        if cf.FROM_CHECKPOINT:
            self.model.load_weights(cf.FROM_CHECKPOINT_PATH)

    def print_summary(self):
        self.model.summary()

    def train(self):
        print("==============================hello")
        csv_logger = CSVLogger(cf.LOG_PATH, append=True, separator=",")

        early_stopping = EarlyStopping(monitor=cf.EARLY_STOPPING_MONITOR,
                                        min_delta=0.0, patience=cf.EARLY_STOPPING_PATIENCE)

        model_checkpoint = ModelCheckpoint(os.path.join(cf.CHECKPOINT_DIR, "weights.{epoch:02d}-{val_loss:.2f}.h5"),
                                                monitor=cf.CHECKPOINT_MONITOR, verbose=0, save_best_only=True,
                                                save_weights_only=False, mode='auto', period=cf.CHECKPOINT_PERIOD)

        history = self.model.fit(self.train_generator(), steps_per_epoch=self.train_generator.steps,
                                epochs=cf.EPOCH, verbose=1, validation_data=self.val_generator(),
                                validation_steps=self.val_generator.steps,
                                shuffle=True,
                                callbacks=[csv_logger, early_stopping, model_checkpoint])

        self.model.save(os.path.join(cf.CHECKPOINT_DIR, "model.h5"))

    def predict(self):
        with open(cf.CSV_PATH_FOR_PREDICT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["data", "true_x", "true_y", "true_z", "pre_x", "pre_y", "pre_z", "px - tx", "py - ty", "pz - tz"])
            self.model.load_weights(cf.MODEL_PATH_FOR_PREDICT)
            for i in range(len(self.test_generator)):
                dir_num, input, label = self.test_generator[i]
                pre_y = self.model.predict(input)
                pre_y = pre_y[0]
                diff = pre_y - label
                writer.writerow([dir_num, label[0], label[1], label[2], pre_y[0], pre_y[1], pre_y[2], diff[0], diff[1], diff[2]])



        