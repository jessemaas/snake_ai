import game
import lstm_ai
import ai as ai_module
import last_n_bodyparts_ai
import convnet_ai

from matplotlib import pyplot
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

import math
import random

food_reward = 1

train_settings = [
#    "teacher",             # whether to learn from a teacher/supervised
    "reinforcement",        # whether to use reinforcement learning
#    "distance_food",       # whether to use the distance_food goal
    "probability_of_food",  # whether to use the probability_of_food goal
]

if False:
    print()
    print('using CPU!')
    print()

    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=16,
        inter_op_parallelism_threads=16, 

        device_count = {'GPU' : 0, 'CPU': 1}
    )

    session = tf.Session(config=config)
    K.set_session(session)

class Trainer:
    def __init__(self, ai, parallel_sessions):
        self.ai = ai
        self.train_data = [[] for i in range(parallel_sessions)]
        self.worlds_with_train_data = [(game.World(), self.train_data[i]) for i in range(parallel_sessions)]

    def step(self):
        """
        performs one simulation step and does necessary work for training
        """

        if "teacher" in train_settings:
            teacher_ai = ai_module.HardcodedAi()
            teacher_move_indices = teacher_ai.predict_best_moves(
                [world for world, train_data in self.worlds_with_train_data]
            )

        # predict moves
        move_indices = self.ai.predict_best_moves(
            [world for world, train_data in self.worlds_with_train_data]
        )

        removed_world_indices = []

        for i in range(len(self.worlds_with_train_data)):
            world, train_data = self.worlds_with_train_data[i]
            world.set_direction(game.directions[move_indices[i]])
            result = world.forward()

            if "distance_food" in train_settings:
                distance_food_x = abs(world.food[0] - world.snake[0][0])
                distance_food_y = abs(world.food[1] - world.snake[0][1])

                distance_food = distance_food_x + distance_food_y
                reward = 0.5 / distance_food if distance_food > 0 else 0.5
                reward = 1.5 * (0.8 ** distance_food)
            elif "reinforcement" in train_settings:
                reward = 0
            elif "teacher" in train_settings:
                reward = 1 if teacher_move_indices[i] == move_indices[i] else 0
                train_data.append(ai_module.LearnData(world, move_indices[i], reward))
            
            reward = 0

            if result == game.MoveResult.death:
                # stop world
                removed_world_indices.append(i)
                if "reinforcement" in train_settings:
                    reward = -1
            elif result == game.MoveResult.eat:
                if "reinforcement" in train_settings:
                    reward = food_reward
                for data in train_data:
                    data.total_food += 1
                    if "probability_of_food" in train_settings and not "teacher" in train_settings:
                        data.reward = 1
            if "probability_of_food" in train_settings and not "teacher" in train_settings:
                train_data.append(ai_module.LearnData(world, move_indices[i], 0))
            elif "reinforcement" in train_settings:
                
                gamma = 0.9
                reward_factor = 1

                if reward != 0:
                    for j in range(len(train_data) - 1, -1, -1):
                        train_data[j].reward += reward * reward_factor
                        reward_factor *= gamma
                
                train_data.append(ai_module.LearnData(world, move_indices[i], reward))

        # remove the larger indices first
        removed_world_indices.sort(reverse=True)

        for index in removed_world_indices:
            del self.worlds_with_train_data[index]
    
    def simulate_entire_game(self):
        while(len(self.worlds_with_train_data) > 0):
            self.step()

    def train(self):
        # flatten train data, TODO: write something I understand myself
        flatten = lambda l: [item for sublist in l for item in sublist]
        flat_train_data = flatten(self.train_data)

        return self.ai.train(flat_train_data)

    def results(self):
        max_score = 0
        total_score = 0

        for datas in self.train_data:
            food_amount = datas[0].total_food

            if max_score < food_amount:
                max_score = food_amount

            total_score += food_amount

        return max_score, total_score
        



# ai = lstm_ai.SimpleAi()
# ai = convnet_ai.CenteredAI("models_output/centered-ai-2020-02-13 19:48:32.244474.h5")
# ai = convnet_ai.CenteredAI('./convnet-ai-2020-02-11 19:46:27.989205.h5')
# ai = convnet_ai.CenteredAI('./convnet-ai-2020-02-11 22:32:29.469784.h5')
# ai = last_n_bodyparts_ai.LastNBodyParts(2)
# ai = ai_module.HardcodedAi()
ai = convnet_ai.CenteredAI()

averages = []
losses = []
smoothed_averages = []

graphic_output_interval = 50
pyplot.figure(0)

epochs = 5000
simultaneous_worlds = 128
simulated_games_count = 0

switch_teacher_to_reinforcement = True
ai.epsilon = 0.05
verbosity = 1

# epsilon_decrement_factor = 0.99


import render

for epoch_id in range(1, epochs + 1):

    # ai.epsilon *= epsilon_decrement_factor
    # print("start epoch")
    # print(epoch_id)
    
    if verbosity >= 2:
        print("creating trainer")
    trainer = Trainer(ai, simultaneous_worlds)
    
    if verbosity >= 2:
        print("simulating game")

    trainer.simulate_entire_game()

    if verbosity >= 2:
        print("get results")
    max_score, total_score = trainer.results()
    
    average = float(total_score) / len(trainer.train_data)

    if verbosity >= 2:
        print('max:', max_score, 'average', average)

    averages.append(average)

    history = trainer.train()
    losses.append(history.history['loss'])

    if epoch_id % graphic_output_interval == 0:
        if verbosity >= 1:
            print('graphic output! Epoch:' + str(epoch_id))
        # ai.print_layer_weights()
        # output graph
        pyplot.style.use('default')
        pyplot.clf()
        pyplot.plot(averages, linewidth=0.5)

        for i in range(epoch_id - graphic_output_interval, epoch_id):
            start_index = max(0, i - 50)
            smoothed_averages.append(sum(averages[start_index : i + 1]) / (i + 1 - start_index))

        pyplot.style.use('seaborn')
        pyplot.plot(smoothed_averages, linewidth=0.5)


        if verbosity == 1:
            print('50-game average', smoothed_averages[-1])

        if switch_teacher_to_reinforcement and smoothed_averages[-1] > 0.6 and "teacher" in train_settings:
            train_settings.remove("teacher")
            train_settings.append(["reinforcement"])
            if verbosity >= 1:
                print() 
                print("switching to reinforcement")
                print()


        pyplot.savefig('graph_output/graph-' + str(epoch_id).rjust(5, '0'), dpi=300)
    
        for _ in range(simulated_games_count):
            renderer = render.Renderer(ai)
            renderer.render_loop()


ai.save()

if False:
    pyplot.figure(0)
    pyplot.plot(averages)
    last_update = epoch_id - (epoch_id % graphic_output_interval)

    for i in range(last_update, epoch_id):
        start_index = max(0, i - 10)
        smoothed_averages.append(sum(averages[start_index : i + 1]) / (i + 1 - start_index))

    pyplot.style.use('seaborn')
    pyplot.plot(smoothed_averages)

    pyplot.show()

if False:
    import render

    for i in range(10):
        renderer = render.Renderer(ai)
    renderer.render_loop()
