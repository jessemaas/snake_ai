import util

total_time = util.start_timer()

import game
import lstm_ai
import ai as ai_module
import last_n_bodyparts_ai
import convnet_ai
import simple_ai

# import render

from matplotlib import pyplot

if util.use_cupy:
    import cupy as np
else:
    import numpy as np
    

import tensorflow as tf
from tensorflow.keras import backend as K

import math
import random
import datetime
import os

from time import perf_counter

food_reward = 1

train_settings = [
#    "teacher",             # whether to learn from a teacher/supervised
    "reinforcement",        # whether to use reinforcement learning
#    "distance_food",       # whether to use the distance_food goal
    "probability_of_food",  # whether to use the probability_of_food goal
#    "dies",                 # whether to keep track of the agent dying the next step
    "probability_next_food",
    "cupy"
]

ai_module.train_settings = train_settings

config = {
    "predict_one_food": True,
    "predict_reduced_reward": True,
    "reward_decrement_factor": 0.96,
}


if False:
    # use cpu
    print()
    print('using CPU!')
    print()
    tf.config.experimental.set_visible_devices([], 'GPU')

class Trainer:
    def __init__(self, ai, parallel_sessions):
        self.ai = ai
        self.train_data = [[] for i in range(parallel_sessions)]
        self.worlds_with_train_data = [(game.World(), self.train_data[i]) for i in range(parallel_sessions)]

    def step(self):
        start_timer = util.start_timer()
        """
        performs one simulation step and does necessary work for training
        """

        # predict moves
        util.times_predicted += 1

        util.predicted_actions += len(self.worlds_with_train_data)

        worlds_only = [world for world, train_data in self.worlds_with_train_data]

        predict_timer = util.start_timer()
        move_indices = self.ai.predict_best_moves(worlds_only)
        util.end_timer(predict_timer, 'predict_best_moves')

        removed_world_indices = []

        for i in range(len(self.worlds_with_train_data)):
            world, train_data = self.worlds_with_train_data[i]
            world.set_direction(game.directions[util.get_if_cupy(move_indices[i])])
            result = world.forward()

            learn_data = ai_module.LearnData(world, move_indices[i], 0)
            train_data.append(learn_data)

            died = False
            if result == game.MoveResult.death:
                # stop world
                removed_world_indices.append(i)
                died = True
            elif result == game.MoveResult.eat:
                learn_data.eat_food = True
            learn_data.died = died
            
        # remove the larger indices first
        removed_world_indices.sort(reverse=True)

        for index in removed_world_indices:
            del self.worlds_with_train_data[index]
        
        util.end_timer(start_timer, 'step')
    
    def simulate_entire_game(self):
        # counter = 0
        old_len = None

        for step_nr in range(max_episodes):
            if len(self.worlds_with_train_data) == 0:
                break

            new_len = len(self.worlds_with_train_data)
            if verbosity >= 2 and old_len != new_len:
                print('step; worlds left =', new_len)
                old_len = new_len
            self.step()

        reward_decrement_factor = ai.config['reward_decrement_factor']
        
        for data_episode in (self.train_data):
            total_food = 0
            reward = 0
            for data_point in reversed(data_episode):
                if data_point.eat_food:
                    total_food += 1
                    reward += 1
                
                data_point.food_from_now_on = 1 if total_food > 0 else 0
                data_point.total_food = total_food
                data_point.reward = reward

                reward *= reward_decrement_factor

        print(len(self.worlds_with_train_data))

        
        if verbosity >= 2:
            print('max steps:', max((len(data) for data in self.train_data)))

    def train(self, epochs=1):
        # flatten train data
        flatten = lambda l: [item for sublist in l for item in sublist]
        flat_train_data = flatten(self.train_data)

        return self.ai.train(flat_train_data, epochs=epochs)

    def results(self):
        max_score = 0
        total_score = 0

        for data_episode in self.train_data:
            food_amount = data_episode[0].total_food

            if max_score < food_amount:
                max_score = food_amount

            total_score += food_amount

        return max_score, total_score
        

def train_supervised(teacher_ai, student_ai, rounds):
        trainer = Trainer(teacher_ai, rounds)
        trainer.simulate_entire_game()
        trainer.ai = student_ai
        trainer.train(3)


averages = []
losses = []
smoothed_averages = []

graphic_output_interval = 10
smooth_average_count = graphic_output_interval

epochs = 1000
max_episodes = 5_000
simultaneous_worlds = 128
simulated_games_count = 0

switch_teacher_to_reinforcement = False

verbosity = 1
initialize_supervised = False
supervised_rounds = 5

best_average = 0
best_model = None


if __name__ == "__main__":
    pyplot.figure(0)
    os.makedirs('./graph_output/', exist_ok=True)
    os.makedirs('./models_output/', exist_ok=True)
    # ai = simple_ai.SimpleAi()
    # ai = ai_module.HardcodedAi()
    # ai = convnet_ai.CenteredAI()
    # ai = convnet_ai.RotatedCenteredAI(train_settings)

    ai = convnet_ai.ConvnetAi(train_settings, config)

    ai.epsilon = 0.07
    min_epsilon = 0.03
    epsilon_decrement_factor = 0.99

    # learning_rate = K.get_value(ai.model.optimizer.lr)
    learning_rate = 0.000003
    min_learning_rate = learning_rate * 0.1
    #learning_rate = min_learning_rate
    learning_rate_decrement_factor = 0.99
    ai.set_learning_rate(learning_rate)

    training_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_graph_name = None
    print('training_start =', training_start)

    done_output = 10

    if initialize_supervised:
        if verbosity >= 1:
            print('training supervised')
        for id in range(supervised_rounds):
            if verbosity >= 1:
                print("supervised round: ", id)
            
            old_verbosity = verbosity
            verbosity = min(verbosity, 1)
            train_supervised(ai_module.HardcodedAi(train_settings), ai, 1024 * 8)
            verbosity = old_verbosity

            if verbosity >= 2:
                print("trained supervised")
            
            if True and verbosity >= 1:
                trainer = Trainer(ai, 1024 * 2)
                trainer.simulate_entire_game()

                max_score, total_score = trainer.results()
                
                average = float(total_score) / len(trainer.train_data)

                print('max:', max_score, 'average', average)

        if False:
            renderer = render.Renderer(ai)
            renderer.render_loop()

    for epoch_id in range(1, epochs + 1):
        start_epoch = util.start_timer()

        # ai.epsilon *= epsilon_decrement_factor
        # print("start epoch")
        if verbosity >= 1:
            print("starting epoch:", epoch_id)
        

        if epoch_id == 1:
            # performs some tests to make sure the GPU works
            name = tf.test.gpu_device_name()
            if name:
                print('Default GPU Device: {}'.format(name))
            else:
                print("not using gpu!")
            
            if tf.test.is_gpu_available():
                print("gpu available")
            else:
                print("no gpu available")


        if verbosity >= 2:
            print("creating trainer")
        trainer = Trainer(ai, simultaneous_worlds)
        
        if verbosity >= 2:
            print("simulating game")

        trainer.simulate_entire_game()

        if verbosity >= 2:
            print("get results")
        max_score, total_score = trainer.results()
        
        # `len(trainer.train_data)` is the amount of simulated worlds
        average = float(total_score) / len(trainer.train_data)

        if verbosity >= 1: # or (initialize_supervised and verbosity == 1 and epoch_id == 1):
            print('max:', max_score, 'average', average)

        averages.append(average)
        if average > 1:
            ai.epsilon = max(ai.epsilon * epsilon_decrement_factor, min_epsilon)

            learning_rate = max(learning_rate * learning_rate_decrement_factor, min_learning_rate)
            ai.set_learning_rate(learning_rate)

        if average > best_average:
            best_average = average
            best_model = tf.keras.models.clone_model(ai.model)

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
                start_index = max(0, i - smooth_average_count)
                smoothed_averages.append(sum(averages[start_index : i + 1]) / (i + 1 - start_index))

            pyplot.style.use('seaborn')
            pyplot.plot(smoothed_averages, linewidth=0.5)

            if verbosity == 1:
                print(str(smooth_average_count) + '-game average:', smoothed_averages[-1])

            if smoothed_averages[-1] > done_output:
                ai.save(training_start, '', '-above-' + str(done_output))
                done_output += 1

            new_graph_name = 'graph_output/graph-' + training_start
            pyplot.savefig(new_graph_name, dpi=300)
            pyplot.clf()

            # Plot the loss
            smoothed_losses = []
            for end in range(len(losses)):
                start = max(0, end - 10)
                total_loss = 0
                for loss_index in range(start, end + 1):
                    total_loss += losses[loss_index][0]
                avg_loss = total_loss / (end - start + 1)

                smoothed_losses.append(avg_loss)

            loss_graph_name = 'graph_output/loss-graph-' + training_start
            pyplot.plot(losses)
            pyplot.plot(smoothed_losses)
            pyplot.savefig(loss_graph_name)
            pyplot.clf()

        
            for _ in range(simulated_games_count):
                renderer = render.Renderer(ai)
                renderer.render_loop()
        util.end_timer(start_epoch, 'epoch')

        util.print_timers()

    ai.save(training_start, '', '-last')
    ai.model = best_model
    ai.save(training_start, '', '-best')

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

util.end_timer(total_time, 'total_time')

util.print_timers()
