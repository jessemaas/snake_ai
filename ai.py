import game
import util

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers

if util.use_cupy:
    import cupy as np
else:
    import numpy as np

from tensorflow.keras import backend as K

import random
import math
import datetime

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
    def __init__(self, train_settings):
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
    def __init__(self, train_settings, epsilon = 0.1):
        super().__init__(train_settings)
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
    def __init__(self, train_settings):
        super().__init__(train_settings)
        self.num_output = 0

        index = 0
        self.dies = "dies" in train_settings
        self.probability_next_food = "probability_next_food" in train_settings
        
        self.reward_index = index
        index += 1

        if self.dies:
            self.dies_index = index
            index += 1

        if self.probability_next_food:
            self.probability_next_food_index = index
            index += 1

        self.num_output = index

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

        start = util.start_timer()
        predictions = self.model.predict(util.as_tensor(inputs))
        util.end_timer(start, 'predictions')

        def direction(world_id):
            # with probability epsilon, return a random action
            if random.random() < self.epsilon:
                return action_indices[world_id * 3 + random.randint(0, 2)]

            max_index = -1
            max_estimate = float("-Infinity")
            

            for i in range(0, 3):
                index = world_id * 3 + i
                prediction = predictions[index]

                if self.num_output >= 2:
                    estimate = prediction[self.reward_index]
                else:
                    estimate = prediction

                if True:
                    action = game.directions[util.get_if_cupy(action_indices[index])]
                                        
                    w = worlds[world_id] 
                    snake_head = w.snake[0]
                    above_snake_head = snake_head[0] + action[0], snake_head[1] + action[1]

                    estimate -= (2 if
                        above_snake_head[0] < 0 or
                        above_snake_head[0] >= w.width or
                        above_snake_head[1] < 0 or
                        above_snake_head[1] >= w.height or
                        above_snake_head in w.snake[1:-1]
                        else 0)
                elif self.dies:
                    estimate -= prediction[self.dies_index] * 0.2
                
                if self.probability_next_food:
                    estimate += prediction[self.probability_next_food_index] * prediction[self.reward_index]

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

        start = util.start_timer()
        train_result = self.model.fit(util.as_tensor(inputs), util.as_tensor(targets), batch_size=512, epochs=epochs, verbose=0)
        util.end_timer(start, 'fitting')
        return train_result
    
    def target_output(self, train_data_list):
        targets = np.empty((len(train_data_list), self.num_output))
        
        for i, data in enumerate(train_data_list):
            targets[i][self.reward_index] = data.reward
            if self.dies:
                targets[i][self.dies_index] = 1 if data.died else 0
            if self.probability_next_food:
                targets[i][self.probability_next_food_index] = 1 if data.total_food >= 2 else 0

        return targets


class EmptyHistory:
    def __init__(self):
        self.history = {"loss": 0}

class AStarAI(BaseAi):
    def __init__(self, train_settings, epsilon = 0.1):
        super().__init__(train_settings)
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
