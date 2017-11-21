import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
######################################
# implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
#
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

# both train_data and valid_data contain tupes of images and labels
# labels are one hot encoded
train_data, train_labels = trans.get_train()
valid_data, valid_labels = trans.get_valid()

#reshape data to represent the history
train_data = train_data.reshape(train_data.shape[0], 4, train_data[0].shape[0]/4)
valid_data = valid_data.reshape(valid_data.shape[0], 4, valid_data[0].shape[0]/4)
#swap axes to get an input of (batch_size, board, history)
np.swapaxes(train_data, 1, 2)
np.swapaxes(valid_data, 1, 2)

print(train_labels.shape)
#Model Parameters
input_shape=train_data.shape[1:]
num_filters=32
filter_size=2
dropout=0.25
pool_size=2
activation='relu'
units=256
dropout_dense=0.5

# Build Keras model
model = Sequential()

model.add(Conv1D(num_filters, filter_size, padding='same', activation=activation, input_shape=input_shape))
model.add(Conv1D(num_filters, filter_size, padding='same', activation=activation))
model.add(MaxPooling1D(pool_size))
model.add(Conv1D(num_filters, filter_size, padding='same', activation=activation))
model.add(Conv1D(num_filters, filter_size, padding='same', activation=activation))
model.add(GlobalMaxPooling1D())
model.add(Dropout(dropout))

model.add(Dense(units, activation=activation))
model.add(Dropout(dropout_dense))
model.add(Dense(units, activation=activation))
model.add(Dropout(dropout_dense))
model.add(Dense(5, activation='softmax'))

#Training Hyperparameters
lr=0.1
decay=1e-6
momentum=0.9
nesterov=True
batch_size=64
epochs=100

#train with SGD
sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(valid_data, valid_labels))

# 2. save your trained model
model.save('agent.hd5')
