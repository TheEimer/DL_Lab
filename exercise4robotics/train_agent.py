import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
from collections import namedtuple

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

Statistics = namedtuple("Stats",["loss", "tests"])

class Agent:
    "Neural Network agent in tensorflow"
    def __init__(self, opt):
        self.model = self.build_model(opt)

    "Build the model"
    def build_model(self, opt):
        self.x = tf.placeholder(tf.float32, shape=[None, opt.state_siz, opt.hist_len, 1])

        # Placeholders for loss
        self.targets = tf.placeholder(tf.float32, shape = [None])
        self.Q_s_next = tf.placeholder(tf.float32, shape=[None, opt.act_num])
        self.action_onehot = tf.placeholder(tf.float32, shape=[None, opt.act_num])
        units = 32

        # Convolutional layers (padded to size)
        self.conv1 = tf.contrib.layers.conv2d(self.x, 32, 2)
        self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 2)
        self.pool1 = tf.contrib.layers.max_pool2d(self.conv2, 2)

        # Flatten before hidden layers
        self.flatten = tf.contrib.layers.flatten(self.pool1)

        # Hidden layers
        self.hidden1 = tf.contrib.layers.fully_connected(self.flatten, 32)
        self.hidden2 = tf.contrib.layers.fully_connected(self.hidden1, 32)

        # Linear output
        self.out = tf.contrib.layers.fully_connected(self.hidden2, opt.act_num, activation_fn=None)

        selected_q = tf.reduce_sum(self.action_onehot * self.out, 1)
        self.loss = tf.losses.mean_squared_error(selected_q, self.targets)

        #Adam optimizer
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(5e-4).minimize(self.loss)

        pass

    def predict(self, sess, states):
        """
        Returns agent prediction
        """
        return sess.run([self.out], feed_dict = {self.x: states})

    def update(self, sess, state_batch, targets, action_batch):
        """
        Trains the network
        """
        self.train_step.run(feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch}, session = sess)
        return sess.run([self.loss], feed_dict={self.x: state_batch, self.targets: targets, self.action_onehot: action_batch})

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

def plot_stats(stats):
    # Plot loss over time
    fig1 = plt.figure(figsize=(10,10))
    plt.plot(stats.loss)
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.title("Loss per step")
    fig1.savefig('loss.png')
    plt.show(fig1)

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
test_sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)
agent = Agent(opt)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if opt.disp_on:
    win_all = None
    win_pob = None

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 3 * 10**3
epi_step = 0
nepisodes = 0
stats = Statistics(loss = np.zeros(steps), tests = np.zeros(steps // 250))

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    # Take action with highest Q-value
    action = np.argmax(agent.predict(sess, state_with_history.T[np.newaxis, ..., np.newaxis]))
    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state

    #Training
    #Get batch
    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
    # get indices of highest Q-values in next states
    next_state_batch = np.reshape(next_state_batch, (opt.minibatch_size, 900, 4))
    state_batch = np.reshape(state_batch, (opt.minibatch_size, 900, 4))[..., np.newaxis]
    next_qs = np.zeros(len(action_batch))
    for i in range(len(action_batch)):
        next_qs[i] = reward_batch[i] + 0.99 * np.max(agent.predict(sess, next_state_batch[i][np.newaxis, ..., np.newaxis])) * (1. - terminal_batch[i])

    # Train agent
    if not np.array_equal([opt.minibatch_size, opt.act_num], action_batch.shape):
        action_batch = trans.one_hot_action(action_batch)

    loss = agent.update(sess, state_batch, next_qs, action_batch)
    stats.loss[step] = loss[0]

    # Print the loss
    print("Loss in step {}: {}" .format(step, loss[0]))


    if step % 250 == 0:
        test_state = test_sim.newGame(opt.tgt_y, opt.tgt_x)
        for i in range(100):
            # check if episode ended
            if state.terminal:
                break
            else:
                #test run
                test_state_with_history = np.zeros((opt.hist_len, opt.state_siz))
                next_test_state = np.copy(test_state_with_history)
                append_to_hist(test_state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
                # Take action with highest Q-value
                action = np.argmax(agent.predict(sess, test_state_with_history.T[np.newaxis, ..., np.newaxis]))
                action_onehot = trans.one_hot_action(action)
                next_state = test_sim.step(action)
                # append to history
                append_to_hist(next_test_state, rgb2gray(next_state.pob).reshape(opt.state_siz))
                # add to the transition table
                trans.add(state_with_history.reshape(-1), action_onehot, next_test_state.reshape(-1), next_state.reward, next_state.terminal)
                # mark next state as current state
                test_state_with_history = np.copy(next_test_state)
                test_state = next_state

    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()


# 2. perform a final test of your model and save it
saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")
plot_stats(stats)

# Test on 500 steps in total

epi_step = 0
nepisodes_test = 0
nepisodes_solved = 0
action = 0

# Restart game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

for step in range(500):

    # Check if episode ended and if yes start new game
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    else:
        action = np.argmax(agent.predict(sess, state_with_history.T[np.newaxis, ..., np.newaxis]))
        state = sim.step(action)
        epi_step += 1

    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()

print("Solved {} of {} episodes" .format(nepisodes_solved, nepisodes_test))
print("Success rate of {}" .format(float(nepisodes_solved) / float(nepisodes_test)))
