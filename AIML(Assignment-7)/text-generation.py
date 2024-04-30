#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("C:\Users\Sanjay Bharadwaj U\Desktop\My cap\AIML(Assignment-7)\game_of_thrones.txt"))


# In[ ]:


# For deleting any pre-existing models
for filename in os.listdir():
    if filename.endswith('.hdf5'):
        os.unlink(filename)
os.listdir()


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np


# In[ ]:


SEQ_LENGTH = 100


# Now that we've imported everything we need form Keras, we're all set to go!
# 
# First, we load our data.

# What np_utils.to_categorical does

# In[ ]:


test_x = np.array([1, 2, 0, 4, 3, 7, 10])

# one hot encoding
test_y = np_utils.to_categorical(test_x)
print(test_x)
print(test_y)


# This functions returns an array of sequences from the input text file and the corresponding output for each sequence encoded as a one-hot vector.

# Now we add a function to create our LSTM.

# In[ ]:


# Using keras functional model
def create_functional_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    mode        = kwargs.get('mode', 'train')
    hidden_dim  = int(hidden_dim)

    inputs      = Input(shape = (input_shape[1], input_shape[2]))
    model       = LSTM(hidden_dim, return_sequences = True)(inputs)
    model       = Dropout(drop)(model)
    model       = Dense(n_out)(model)


# In[ ]:


# Using keras sequential model
def create_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    mode        = kwargs.get('mode', 'train')
    hidden_dim  = int(hidden_dim)
    model       = Sequential()
    flag        = True 

    if n_layers == 1:   
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2])) )
        if mode == 'train':
            model.add( Dropout(drop) )

    else:
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2]), return_sequences = True) )
        if mode == 'train':
            model.add( Dropout(drop) )
        for i in range(n_layers - 2):
            model.add( LSTM(hidden_dim, return_sequences = True) )
            if mode == 'train':
                model.add( Dropout(drop) )
        model.add( LSTM(hidden_dim) )

    model.add( Dense(n_out, activation = activ) )

    return model


# Training our model

# In[ ]:


def train(model, X, Y, n_epochs, b_size, vocab_size, **kwargs):    
    loss            = kwargs.get('loss', 'categorical_crossentropy')
    opt             = kwargs.get('optimizer', 'adam')
    
    model.compile(loss = loss, optimizer = opt)

    filepath        = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint      = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list  = [checkpoint]
    X               = X / float(vocab_size)
    model.fit(X, Y, epochs = n_epochs, batch_size = b_size, callbacks = callbacks_list)


# The fit function will run the input batchwase n_epochs number of times and it will save the weights to a file whenever there is an improvement. This is taken care of through the callback. <br><br>
# After the training is done or once you find a loss that you are happy with, you can test how well the model generates text.

# In[ ]:


def generate_text(model, X, filename, ix_to_char, vocab_size):
    
    # Load the weights from the epoch with the least loss
    model.load_weights(filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    start   = np.random.randint(0, len(X) - 1)
    pattern = np.ravel(X[start]).tolist()

    # We seed the model with a random sequence of 100 so it can start predicting
    print ("Seed:")
    print ("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
    output = []
    for i in range(250):
        x           = np.reshape(pattern, (1, len(pattern), 1))
        x           = x / float(vocab_size)
        prediction  = model.predict(x, verbose = 0)
        index       = np.argmax(prediction)
        result      = index
        output.append(result)
        pattern.append(index)
        pattern = pattern[1 : len(pattern)]

    print("Predictions")
    print ("\"", ''.join([ix_to_char[value] for value in output]), "\"")


# Now we're ready to either train or test our model.

# In[ ]:


filename    = '../input/game_of_thrones.txt'
data        = open(filename).read()
data        = data.lower()
# Find all the unique characters
chars       = sorted(list(set(data)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
ix_to_char  = dict((i, c) for i, c in enumerate(chars))
vocab_size  = len(chars)

print("List of unique characters : \n", chars)

print("Number of unique characters : \n", vocab_size)

print("Character to integer mapping : \n", char_to_int)


# In[ ]:


list_X      = []
list_Y      = []

# Python append is faster than numpy append. Try it!
for i in range(0, len(data) - SEQ_LENGTH, 1):
    seq_in  = data[i : i + SEQ_LENGTH]
    seq_out = data[i + SEQ_LENGTH]
    list_X.append([char_to_int[char] for char in seq_in])
    list_Y.append(char_to_int[seq_out])

n_patterns  = len(list_X)
print("Number of sequences in data set : \n", n_patterns)
print(list_X[0])
print(list_X[1])


# In[ ]:


X           = np.reshape(list_X, (n_patterns, SEQ_LENGTH, 1)) # (n, 100, 1)
# Encode output as one-hot vector
Y           = np_utils.to_categorical(list_Y)

print(X[0])
print(Y[0])


# In[ ]:


print("Shape of input data ", X.shape, "\nShape of output data ", Y.shape)


# In[ ]:


model   = create_model(1, X.shape, 256, Y.shape[1], mode = 'train')


# In[ ]:


train(model, X[:1024], Y[:1024], 2, 512, vocab_size)


# In[ ]:


# Iterating through each model and generating text
for filename in os.listdir():
    if filename.endswith('.hdf5'):
        print("Model Name:", filename)
        generate_text(model, X, filename, ix_to_char, vocab_size)

