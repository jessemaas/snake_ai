import ai
import game


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
import numpy as np
#from keras.backend import tf
from tensorflow.keras.backend import set_session

import datetime

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

tile_classes = 4

class ConvnetAi(ai.BaseAi):
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(game.world_width, game.world_height, tile_classes)))
        model.add(layers.Conv2D(6, (5, 5), activation='relu'))
        #model.add(layers.Conv2D(4, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(ai.direction_count, activation='relu'))

        self.model = model
        optimizer = keras.optimizers.SGD()
        model.compile(optimizer=optimizer, loss=ai.custom_loss)
        
        self.epsilon = 0.10

    def worlds_to_np_array(self, worlds):
        result = np.zeros((len(worlds), game.world_width, game.world_height, tile_classes), dtype=np.float)
        
        for world_index, world in enumerate(worlds):
            food_x, food_y = world.food

            result[world_index, food_x, food_y, 1] = 1

            for snake_x, snake_y in world.snake:
                result[world_index, snake_x, snake_y, 2] = 1

            head_x, head_y = world.snake[0]
            result[world_index, head_x, head_y, 3] = 1

            for x in range(game.world_width):
                for y in range(game.world_height):
                    if np.sum(result[world_index, x, y]) == 0:
                        result[world_index, x, y, 0] = 1

        return result

    def save(self):
        self.model.save('convnet-ai-' +  str(datetime.datetime.now()) + '.h5')

class CenteredAI(ai.BaseAi):
    def __init__(self, save_file=None):
        if save_file == None:
            model = models.Sequential()
            model.add(layers.Conv2D(8, (3, 3), input_shape=(game.world_width * 2 - 1, game.world_height * 2 - 1, tile_classes)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.PReLU())
            model.add(layers.Conv2D(6, (5, 5)))
            #model.add(layers.Conv2D(4, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.PReLU())
            model.add(layers.Dense(ai.direction_count))
            if True:
                model.add(layers.Dense(ai.direction_count, activation="sigmoid"))
            else:
                model.add(layers.Dense(ai.direction_count))
                model.add(layers.PReLU())

            self.model = model
            optimizer = keras.optimizers.SGD()
            model.compile(optimizer=optimizer, loss=ai.custom_loss)
        else:
            self.model = load_model(save_file, custom_objects={'custom_loss': ai.custom_loss})
        
        self.epsilon = 0.05

    def worlds_to_np_array(self, worlds):
        result = np.zeros((len(worlds), game.world_width * 2 - 1, game.world_height * 2 - 1, tile_classes), dtype=np.float)
        result[:, :, :, 2] = 1

        for world_index, world in enumerate(worlds):
            food_x, food_y = world.food
            head_x, head_y = world.snake[0]
            
            x_offset = -head_x + game.world_width - 1
            y_offset = -head_y + game.world_height - 1

            result[world_index, head_x:head_x + game.world_width, head_y:head_y + game.world_height] = [0, 0, 0, 1]

            """
            for x in range(result.shape[1]):
                for y in range(result.shape[2]):
                    # there is no food and part of the snake here
                    if x_offset + x < 0 or x_offset + x >= game.world_width or y_offset + y < 0 or y_offset + y >= game.world_height:
                        # the tile is outside the world border
                        result[world_index, x, y, 2] = 1
                    else:
                        # the tile is inside the world border
                        result[world_index, x, y, 3] = 1
            """

            result[world_index, food_x + x_offset, food_y + y_offset] = np.array([1, 0, 0, 0])

            for snake_x, snake_y in world.snake:
                result[world_index, snake_x + x_offset, snake_y + y_offset] = np.array([0, 1, 0, 0])

        return result

    def save(self):
        self.model.save('convnet-ai-' +  str(datetime.datetime.now()) + '.h5')
        
