import tensorflow.keras as keras
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras import backend as K

import game
# import train

import random
import math
import datetime

train_settings = None

sign = lambda x: x and (1, -1)[x < 0]

class LearnData:
    def __init__(self, world, action_index, reward, died=None):
        self.width = world.width
        self.height = world.height
        self.snake = world.snake[:]
        self.food = world.food
        self.action_index = action_index
        self.reward = reward
        self.total_food = 0
        self.died = died

tile_class_count = 3
direction_count = 4
world_dimension_count = 2
max_snake_len = 20

def custom_loss(y_true, y_pred):
    diff = y_true - y_pred
    max_diff = K.max(diff, axis=1)
    return K.square(max_diff)

class BaseAi:
    def __init__(self):
        self.epsilon = 0.05
        self.model = None
        self.policy = self.simple_policy

    def simple_policy(self, prediction):
        max_index = 0

        for i in range(1, direction_count):
            if prediction[i] > prediction[max_index]:
                max_index = i

        return max_index

    def predict_best_moves(self, worlds):
        if random.random() < self.epsilon:
            return [random.randint(0, direction_count - 1) for world in worlds]
        else:
            inputs = self.worlds_to_np_array(worlds)
            predictions = self.model.predict(inputs)

            return [self.policy(prediction) for prediction in predictions]

    def worlds_to_np_array(self, worlds):
        raise NotImplementedError

    def set_learning_rate(self, learning_rate):
        K.set_value(self.model.optimizer.lr, learning_rate)

    def train(self, learnData, epochs=1):
        inputs = self.worlds_to_np_array(learnData)
        targets = self.target_output(learnData)

        return self.model.fit(inputs, targets, batch_size=512, epochs=epochs, verbose=0)

    def print_layer_weights(self):
        for layer in self.model.layers:
            print(layer.get_weights()) 

    def save(self, time, prefix='', suffix=''):
        self.model.save('./models_output/' + prefix +  str(time) + suffix + '.h5')

    def target_output(self, train_data_list):
        targets = np.empty((len(train_data_list), direction_count))
        targets.fill(-np.Infinity)

        for i, data in enumerate(train_data_list):
            targets[i][data.action_index] = data.reward

        return targets

class HardcodedAi(BaseAi):
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def predict_best_moves(self, worlds):
        move_directions = []
        
        for world in worlds:
            if random.random() < self.epsilon:
                move_directions.append(game.random_direction())
            else:
                distance_x = world.food[0] - world.snake[0][0]
                distance_y = world.food[1] - world.snake[0][1]
                abs_x = abs(distance_x)
                abs_y = abs(distance_y)

                if abs_x == abs_y:
                    if distance_x == 0:
                        move_directions.append((1, 0))
                    else:
                        move_directions.append((sign(distance_x), 0))
                elif(abs_x > abs_y):
                    move_directions.append((sign(distance_x), 0))
                else:
                    move_directions.append((0, sign(distance_y)))

        return list(map(lambda pos: game.directions.index(pos), move_directions))


    def train(self, learnData):
        return EmptyHistory()

    def save(self, time, prefix='', suffix=''):
        pass

class RotatedAI(BaseAi):            
    def predict_best_moves(self, worlds):
        unrotated_inputs = self.worlds_to_np_array(worlds)
        shape = unrotated_inputs.shape
        inputs = np.empty((shape[0] * 3,) + shape[1:])
        action_indices = np.empty(shape[0] * 3, dtype=int)

        for world_id in range(len(worlds)):
            i = 0
            for action_index in range(4):
                previous_direction = worlds[world_id].snake_direction

                if game.directions[action_index] != (-previous_direction[0], -previous_direction[1]):
                    index = world_id * 3 + i

                    inputs[index] = self.rotate(unrotated_inputs[world_id], action_index)
                    action_indices[index] = action_index
                    i += 1

        predictions = self.model.predict(inputs)

        def direction(world_id):
            # with probability epsilon, return a random action
            if random.random() < self.epsilon:
                return action_indices[world_id * 3 + random.randint(0, 2)]

            max_index = -1
            max_estimate = float("-Infinity")

            for i in range(0, 3):
                index = world_id * 3 + i

                if True:
                    action = game.directions[action_indices[index]]
                                        
                    w = worlds[world_id] 
                    snake_head = w.snake[0]
                    above_snake_head = snake_head[0] + action[0], snake_head[1] + action[1]

                    # print(above)
                    # estimate = predictions[index][0] - (2 if above[3] != 1 and above[0] != 1 else 0)
                    estimate = predictions[index][0] - (2 if
                        above_snake_head[0] < 0 or
                        above_snake_head[0] >= w.width or
                        above_snake_head[1] < 0 or
                        above_snake_head[1] >= w.height or
                        above_snake_head in w.snake[1:-1]
                        else 0)
                elif "dies" in train_settings:
                    estimate = predictions[index][0] - predictions[index][1] * 0.2# - (2 if inputs[index, game.world_width - 1, game.world_height, 3] != 1 else 0)
                else:
                    estimate = predictions[index]

                if estimate > max_estimate:
                    max_index = index
                    max_estimate = estimate

            return action_indices[max_index]


        return [direction(world_id) for world_id in range(len(worlds))]

    def rotate(self, world_as_np_array, amount):
        raise NotImplementedError()

    def train(self, learnData, epochs=1):
        inputs = self.worlds_to_np_array(learnData)

        for i, data in enumerate(learnData):
            inputs[i] = self.rotate(inputs[i], data.action_index)
        
        targets = self.target_output(learnData)

        return self.model.fit(inputs, targets, batch_size=512, epochs=epochs, verbose=0)
    
    def target_output(self, train_data_list):
        if "dies" in train_settings:
            targets = np.empty((len(train_data_list), 2))

            for i, data in enumerate(train_data_list):
                targets[i][0] = data.reward
                targets[i][1] = 1 if data.died else 0
        else:
            targets = np.empty((len(train_data_list), 1))

            for i, data in enumerate(train_data_list):
                targets[i][0] = data.reward

        return targets


class EmptyHistory:
    def __init__(self):
        self.history = {"loss": 0}

class AStarAI(BaseAi):
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def predict_best_moves(self, worlds):
        move_directions = []
        
        for world in worlds:
            if random.random() < self.epsilon:
                move_directions.append(game.random_direction())
            else:
                raise NotImplementedError()

        return list(map(lambda pos: game.directions.index(pos), move_directions))

    def heuristic(self, food_location, agent_location):
        distance_x = food_location[0] - agent_location[0][0]
        distance_y = food_location[1] - agent_location[0][1]
        return abs(distance_x) + abs(distance_y)


    def train(self, learnData):
        return EmptyHistory()

    def save(self, time, prefix='', suffix=''):
        pass