
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import re, os, random
import random
random.seed(102)

max_samples = 1000000000000
dwidth, ddepth = 13, 300
aupath = '/work/ssd/projects/word_level/words_48_384_gray_npz/'

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Reshape, Permute, Concatenate, Activation
from keras.layers import LSTM, TimeDistributed, CuDNNLSTM, Flatten, Conv1D, Lambda, concatenate, BatchNormalization, Bidirectional
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.utils import multi_gpu_model as MGM
import tensorflow as tf
from keras import backend as K
from batchGen import DataGenerator
from keras.applications import inception_v3
#from utils import layer_utils
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

random.seed(101)

batch_size=32
train_in_file_path = "/work/sentence_type_classification/speech_classification/train_dataset/"
val_in_file_path = "/work/sentence_type_classification/speech_classification/validation_dataset/"

train_generator = DataGenerator(train_in_file_path, batch_size=batch_size, dwidth=dwidth, ddepth=ddepth, other_class_cnt=25000)
val_generator = DataGenerator(val_in_file_path, batch_size=batch_size, dwidth=dwidth, ddepth=ddepth, other_class_cnt=2500)

print("creating Model")
print("---------------")
input_mfcc = Input(shape=(ddepth, dwidth))
### Model1
'''
x = CuDNNLSTM(26*40, return_sequences=False)(input_mfcc)
x = Reshape( (40, 26), name='rshp_1' ) (x)
x = CuDNNLSTM(52*20, return_sequences=False)(x)
x = Reshape( (20, 52), name='rshp_2' ) (x)
x = CuDNNLSTM(300*5, return_sequences=False)(x)
x = Dense(512, activation='relu', name='dense_1')(x)
'''




### Model2
x = CuDNNLSTM(13, return_sequences=True)(input_mfcc)
x = CuDNNLSTM(26*80, return_sequences=False)(x)
x = Reshape( (80, 26), name='rshp_1' ) (x)
x = CuDNNLSTM(52*40, return_sequences=False)(x)
x = Reshape( (40, 52), name='rshp_2' ) (x)
x = CuDNNLSTM(300*15, return_sequences=False)(x)
x = Dense(4096, activation='relu', name='dense_1')(x)
x = Dense(2048, activation='relu', name='dense_2')(x)
x = Dense(1024, activation='relu', name='dense_3')(x)

x = Dense(3, activation='softmax', name='output')(x)
output = x

classifier_model = Model(input_mfcc, output)

classifier_model.summary()
plot_model(classifier_model, to_file='speech_classifier_rnn_only.png', show_shapes=True)

print("Compiling Model")
print("---------------")
classifier_model.compile(optimizer='SGD', loss="categorical_crossentropy", metrics=['accuracy'])

print("Training Model")
print("--------------")
for i in range(1, 1000):
    print('Iteration', str(i))
    classifier_model.fit_generator(train_generator, validation_data=val_generator, epochs=1,
        use_multiprocessing=True, workers=6, max_queue_size=18, shuffle=True, verbose=1)
    classifier_model.save('models/speech_classification_'+str(i)+'.hf5')

