import tensorflow.keras as keras
from tensorflow.keras import models, layers
import numpy as np
from functools import reduce

import random

import ai
import datetime


class LSTMAi(ai.BaseAi):
    def __init__(self, file=None):
        if file == None:
            model = models.Sequential()
            model.add(layers.Masking(mask_value=np.NaN, input_shape=(ai.max_snake_len, ai.max_snake_len * 2)))
            model.add(layers.LSTM(32, input_shape=(None, ai.max_snake_len * 2)))
            model.add(layers.Dense(ai.direction_count, activation=None))

            self.model = model

            optimizer = keras.optimizers.RMSprop()
            model.compile(optimizer=optimizer, loss='mse')

        else:
            self.model = keras.models.load_model(file)
        self.epsilon = 0.05

    def worlds_to_np_array(self, worlds):
        # max_snake_len = reduce(lambda max, current_world: len(current_world.snake) if len(current_world.snake) > max else max, worlds, 0)
        result = np.empty((len(worlds), ai.max_snake_len, ai.max_snake_len * 2), dtype=np.float)

        for world_index, world in enumerate(worlds):
            food_x, food_y = world.food

            for i in range(len(world.snake)):
                x, y = world.snake[i]
                result[world_index, i, 0] = x
                result[world_index, i, 1] = y
                result[world_index, i, 2] = food_x
                result[world_index, i, 3] = food_y
        
        return result

    def train(self, learnData):
        inputs = self.worlds_to_np_array(learnData)

        # array = np.zeros((len(inputs), 3, 4), dtype=np.int)
        # for x in range(len(inputs)):
        #     for y in range(3):
        #         for z in range(4):
        #             array[x, y, z] = 0 #targets[x][0][y][z]
        # inputs = array

        targets = np.empty((len(learnData), ai.direction_count))
        targets.fill(-1)
        for i, data in enumerate(learnData):
            targets[i][data.action_index] = data.reward

        return self.model.fit(inputs, targets, batch_size=128)

    def save(self):
        self.model.save('lstm-ai-' +  str(datetime.datetime.now()) + '.h5')

    