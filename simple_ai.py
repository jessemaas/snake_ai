import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
import numpy as np

import random

import ai
import datetime

class SimpleAi(ai.BaseAi):
    def __init__(self, file=None):
        if file == None:
            model = models.Sequential()

            if True:
                model.add(layers.Dense(6, input_shape=(4,), activation='relu'))
                model.add(layers.Dense(ai.direction_count, activation=None))
            elif True:
                model.add(layers.Dense(ai.direction_count, input_shape=(4,)))

            self.model = model

            optimizer = keras.optimizers.SGD(lr=0.01)
            model.compile(optimizer=optimizer, loss=ai.custom_loss)
        else:
            self.model = keras.models.load_model(file)

        self.epsilon = 0.10

    def worlds_to_np_array(self, worlds):
        result = np.empty((len(worlds), ai.world_dimension_count * 2), dtype=np.float)

        for world_index, world in enumerate(worlds):
            food_x, food_y = world.food
            tail_x, tail_y = world.snake[0]
            
            result[world_index, 0] = tail_x
            result[world_index, 1] = tail_y
            result[world_index, 2] = food_x
            result[world_index, 3] = food_y
        
        return result
    
    def save(self, time, prefix='', suffix=''):
        super().save(time, prefix + 'simple-ai-', suffix)

class First2BodyPartsAI(ai.BaseAi):
    def __init__(self, file=None):
        self.body_parts = 2

        if file == None:
            model = models.Sequential()

            if True:
                model.add(layers.Dense(6, input_shape=(2 + 2 * self.body_parts,), activation='relu'))
                model.add(layers.Dense(ai.direction_count, activation=None))
            elif True:
                model.add(layers.Dense(ai.direction_count, input_shape=(4,)))

            self.model = model

            optimizer = keras.optimizers.SGD(lr=0.01)
            model.compile(optimizer=optimizer, loss=ai.custom_loss)            
        else:
            self.model = load_model(file, custom_objects={'custom_loss': ai.custom_loss})
        
        self.epsilon = 0.05
