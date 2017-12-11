import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
# both train_data and valid_data contain tupes of images and labels
# labels are one hot encoded
train_data, train_labels = trans.get_train()
valid_data, valid_labels = trans.get_valid()

#reshape data to represent the history and get the screen version
train_data = train_data.reshape(train_data.shape[0], 25, 25, 4)
valid_data = valid_data.reshape(valid_data.shape[0], 25, 25, 4)

#Model Parameters
input_shape=train_data.shape[1:]
num_filters_first=32
num_filters_second=64
filter_size=3
filter_size_second=3
dropout_conv=0.25
pool_size=2
activation='relu'
units=128
dropout_dense=0.5

# Build Keras model
model = Sequential()

model.add(Conv2D(num_filters_first, filter_size, padding='same', activation=activation, input_shape=input_shape))
model.add(Conv2D(num_filters_first, filter_size, padding='same', activation=activation))
model.add(MaxPooling2D())
model.add(Dropout(dropout_conv))

model.add(Conv2D(num_filters_second, filter_size_second, padding='same', activation=activation, input_shape=input_shape))
model.add(Conv2D(num_filters_second, filter_size_second, padding='same', activation=activation))
model.add(GlobalMaxPooling2D())
model.add(Dropout(dropout_conv))

model.add(Dense(units, activation=activation))
model.add(Dense(units, activation=activation))
model.add(Dense(units, activation=activation))
model.add(Dense(units, activation=activation))
model.add(Dropout(dropout_dense))
model.add(Dense(5, activation='softmax'))

#Training Hyperparameters
lr=0.01
decay=1e-6
momentum=0.9
nesterov=True
batch_size=32
epochs=10

#train with SGD
sgd = SGD(lr=lr, clipnorm=1., decay=decay, momentum=momentum, nesterov=nesterov)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(valid_data, valid_labels))

# 2. save your trained model
model.save('agent.hd5')
