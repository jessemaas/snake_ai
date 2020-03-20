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

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
config.gpu_options.force_gpu_compatible = True

set_session(tf.Session(config=config))

tile_classes = 4

class ConvnetAi(ai.BaseAi):
    def __init__(self, save_file=None):
        self.outside_world_margin = 1
        if save_file == None:
            double_margin = self.outside_world_margin * 2

            model = models.Sequential()
            model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(game.world_width + double_margin, game.world_height + double_margin, 5)))
            model.add(layers.Conv2D(6, (3, 3), activation='relu'))
            model.add(layers.Conv2D(6, (3, 3), activation='relu'))
            model.add(layers.Conv2D(4, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(ai.direction_count, activation='relu'))

            self.model = model
            optimizer = keras.optimizers.SGD()
            model.compile(optimizer=optimizer, loss=ai.custom_loss)
        else:
            # read model from file
            self.model = load_model(save_file, custom_objects={'custom_loss': ai.custom_loss})
        
        self.epsilon = 0.10

    def worlds_to_np_array(self, worlds):
        double_margin = self.outside_world_margin * 2
        result = np.zeros((len(worlds), game.world_width + double_margin, game.world_height + double_margin, 5), dtype=np.float)
        
        for world_index, world in enumerate(worlds):
            food =          np.array([1, 0, 0, 0, 0])
            snake_tail =    np.array([0, 1, 0, 0, 0])
            snake_head =    np.array([0, 0, 1, 0, 0])
            empty =         np.array([0, 0, 0, 1, 0])
            outside_world = np.array([0, 0, 0, 0, 1])

            result[world_index, :, :] = outside_world
            result[world_index, self.outside_world_margin:-self.outside_world_margin, self.outside_world_margin:-self.outside_world_margin] = empty

            food_x, food_y = world.food

            result[world_index, food_x, food_y] = food

            for snake_x, snake_y in world.snake:
                result[world_index, snake_x, snake_y] = snake_tail

            head_x, head_y = world.snake[0]
            result[world_index, head_x, head_y] = snake_head

        return result

    def save(self, time, prefix='', suffix=''):
        super().save(time, prefix + 'convnet-ai-', suffix)

class CenteredAI(ai.BaseAi):
    """
    This AI is a convolutional neural network. It has as input a map of the world, centered around the head of the snake.
    """
    def __init__(self, save_file=None):
        self.tile_classes = 5

        if save_file == None:
            # construct model
            model = models.Sequential()
            model.add(layers.Conv2D(8, (3, 3), input_shape=(game.world_width * 2 - 1, game.world_height * 2 - 1, self.tile_classes)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.PReLU())

            if True:
                # a more complex model
                model.add(layers.Conv2D(6, (3, 3)))
                model.add(layers.PReLU())
                model.add(layers.Conv2D(4, (3, 3)))
            else:
                # a simpler model
                model.add(layers.Conv2D(6, (5, 5)))

            model.add(layers.Flatten())
            model.add(layers.PReLU())
            model.add(layers.Dense(ai.direction_count))
            model.add(layers.PReLU())

            if True:
                # used for "teacher" goal and "food_probabilty" goal
                model.add(layers.Dense(ai.direction_count, activation="sigmoid"))
            else:
                # used for predicting the reward
                model.add(layers.Dense(ai.direction_count, activation='relu'))
            

            self.model = model
            optimizer = keras.optimizers.Adam()
            model.compile(optimizer=optimizer, loss=ai.custom_loss)
        else:
            # read model from file
            self.model = load_model(save_file, custom_objects={'custom_loss': ai.custom_loss})
        
        self.epsilon = 0.05

    def worlds_to_np_array(self, worlds):
        """
        Converts the worlds to into 2D maps, centered around the head of the snake.
        """
        food_encoding =         np.array([1, 0, 0, 0, 0])
        snake_tail_encoding =   np.array([0, 1, 0, 0, 0])
        snake_head_encoding =   np.array([0, 0, 1, 0, 0])
        empty_encoding =        np.array([0, 0, 0, 1, 0])

        # initialize all tiles to "outside the borders"
        result = np.zeros((len(worlds), game.world_width * 2 - 1, game.world_height * 2 - 1, self.tile_classes), dtype=np.float)
        result[:, :, :, 4] = 1


        for world_index, world in enumerate(worlds):
            food_x, food_y = world.food
            head_x, head_y = world.snake[0]
            
            x_offset = -head_x + game.world_width - 1
            y_offset = -head_y + game.world_height - 1

            # set the right tiles to "inside the borders, but empty"
            result[world_index, head_x:head_x + game.world_width, head_y:head_y + game.world_height] = empty_encoding

            # set the right tile to "food"
            result[world_index, food_x + x_offset, food_y + y_offset] = food_encoding

            # set right tiles to "snake"
            for snake_x, snake_y in world.snake:
                result[world_index, snake_x + x_offset, snake_y + y_offset] = snake_tail_encoding
                
            result[world_index, head_x + x_offset, head_y + y_offset] = snake_head_encoding

        return result


    def save(self, time, prefix='', suffix=''):
        super().save(time, prefix + 'centered-ai-', suffix)
        

class RotatedCenteredAI(ai.RotatedAI):
    def __init__(self, save_file=None):
        self.tile_classes = 5
        self.epsilon = 0.01
        self.num_output = 2 if "dies" in ai.train_settings else 1

        if save_file == None:
            # construct model
            model = models.Sequential()
            model.add(layers.Conv2D(8, (5, 5), input_shape=(game.world_width * 2 - 1, game.world_height * 2 - 1, self.tile_classes)))
            model.add(layers.PReLU())

            complex_model = True

            if complex_model:
                # a more complex model
                model.add(layers.Conv2D(6, (3, 3)))
                model.add(layers.PReLU())
                model.add(layers.Conv2D(6, (3, 3)))
                model.add(layers.PReLU())
                # model.add(layers.Conv2D(6, (5, 5), padding='same',))
                model.add(layers.Conv2D(6, (3, 3)))
            else:
                # a simpler model
                model.add(layers.Conv2D(6, (3, 3)))

            model.add(layers.Flatten())
            model.add(layers.PReLU())
            if complex_model:
                model.add(layers.Dense(32))
                model.add(layers.PReLU())

            if True:
                # used for "teacher" goal and "food_probabilty" goal
                model.add(layers.Dense(self.num_output, activation="sigmoid"))
            else:
                # used for predicting the reward
                model.add(layers.Dense(self.num_output, activation='relu'))
            

            self.model = model
            optimizer = keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
        else:
            # read model from file
            self.model = load_model(save_file, custom_objects={'custom_loss': ai.custom_loss})

    # def worlds_to_np_array(self, worlds):
    #     unrotated = CenteredAI.worlds_to_np_array(self, worlds)
    #     result = np.zeros((len(worlds) * 4, game.world_width * 2 - 1, game.world_height * 2 - 1, self.tile_classes), dtype=np.float)
        
    #     for world_id in range(len(worlds)):
    #         for i in range(4):
    #             result[world_id * 4 + i] = np.rot90(unrotated[world_id], i, (0, 1))

    #     return result

    worlds_to_np_array = CenteredAI.worlds_to_np_array

    def rotate(self, world_as_np_array, amount):
        return np.rot90(world_as_np_array, amount, (0, 1))

    def save(self, time, prefix='', suffix=''):
        super().save(time, prefix + 'centered-rotated-ai-', suffix)