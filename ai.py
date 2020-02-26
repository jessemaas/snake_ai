import tensorflow.keras as keras
from tensorflow.keras import models, layers
import numpy as np
import random
from tensorflow.keras import backend as K
import math
import game
import datetime

sign = lambda x: x and (1, -1)[x < 0]

class LearnData:
    def __init__(self, world, action_index, reward):
        self.width = world.width
        self.height = world.height
        self.snake = world.snake[:]
        self.food = world.food
        self.action_index = action_index
        self.reward = reward
        self.total_food = 0

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

    def predict_best_moves(self, worlds):
        if random.random() < self.epsilon:
            return [random.randint(0, direction_count - 1) for world in worlds]
        else:
            inputs = self.worlds_to_np_array(worlds)
            predictions = self.model.predict(inputs)

            def direction(prediction):
                max_index = 0

                for i in range(1, direction_count):
                    if prediction[i] > prediction[max_index]:
                        max_index = i

                return max_index

                # minimum = 0
                # for i in range(0, direction_count):
                #     if prediction[i] < minimum:
                #         minimum = prediction[i]

                # prediction_sum = 0
                # for i in range(0, direction_count):
                #     prediction_sum += prediction[i] - minimum

                # rand = random.random() * prediction_sum - 0.01
                
                # #print('rand before:', rand)

                # for i in range(0, direction_count):
                #     rand -= prediction[i] - minimum
                #     if(rand <= 0):
                #         return i

                # print('rand after:', rand)


            return [direction(prediction) for prediction in predictions]

    def worlds_to_np_array(self, worlds):
        raise NotImplementedError

    def set_learning_rate(self, learning_rate):
        K.set_value(self.model.optimizer.lr, learning_rate)

    def train(self, learnData, epochs=1):
        inputs = self.worlds_to_np_array(learnData)

        targets = np.empty((len(learnData), direction_count))
        targets.fill(-np.Infinity)
        for i, data in enumerate(learnData):
            targets[i][data.action_index] = data.reward

        return self.model.fit(inputs, targets, batch_size=512, epochs=epochs, verbose=0)

    def print_layer_weights(self):
        for layer in self.model.layers:
            print(layer.get_weights()) 

    def save(self, prefix='', suffix=''):
        self.model.save('./models_output/' + prefix +  str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + suffix + '.h5')

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

    def save(self, prefix='', suffix=''):
        pass


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

    def save(self, prefix='', suffix=''):
        pass