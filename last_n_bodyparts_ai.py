import ai
import tensorflow.keras as keras
from tensorflow.keras import models, layers
import numpy as np

import datetime

class LastNBodyParts(ai.BaseAi):
    def __init__(self, n):
        model = models.Sequential()
        # model.add(layers.Dense(ai.direction_count, input_shape=(2 * n + 2,)))
        model.add(layers.Dense(16, input_shape=(2 * n + 2,), activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(ai.direction_count))


        self.model = model
        self.n = n
        optimizer = keras.optimizers.SGD()
        model.compile(optimizer=optimizer, loss=ai.custom_loss)
        
        self.epsilon = 0.10

    def worlds_to_np_array(self, worlds):
        result = np.empty((len(worlds), 2 * self.n + 2), dtype=np.float)

        for world_index, world in enumerate(worlds):
            food_x, food_y = world.food

            result[world_index, 0] = food_x
            result[world_index, 1] = food_y

            for i in range(self.n):
                x, y = world.snake[i]#[-i - 1]
                result[world_index, i * 2 + 2] = x
                result[world_index, i * 2 + 3] = y

        return result

    def save(self):
        self.model.save('last-n-body-parts-ai-' +  str(datetime.datetime.now()) + '.h5')

