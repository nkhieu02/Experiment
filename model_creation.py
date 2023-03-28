import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers
import numpy as np
from KrausLayer import KrausLayer

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)

def create_mvl_model(input_shape, output_shape):
    model: keras.Sequential = keras.models.Sequential()
    if len(input_shape) > 1:
        model.add(layers.Flatten(input_shape=[20,20]))
    else:
        model.add(layers.Input(input_shape))
    model.add(layers.Dense(output_shape))
    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam())
    return model

def create_rr_model(input_shape, output_shape, rank):
    model: keras.Sequential = keras.Sequential()
    if len(input_shape) > 1:
        model.add(layers.Flatten(input_shape=input_shape))
    else:
        model.add(layers.Input(input_shape))
    model.add(layers.Dense(units = rank))
    model.add(layers.Dense(units = output_shape))
    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam())
    return model

def create_trace_model(input_shape, rank):
    model = keras.Sequential()
    model.add(KrausLayer(1, rank, input_shape))
    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(0.001))

    return model

