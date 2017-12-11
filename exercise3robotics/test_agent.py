import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Sequential
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

agent = load_model('agent.hd5')
input_hist_flat = np.zeros([1, 25*25*4])

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.eval_steps):

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        input_hist_flat = np.roll(input_hist_flat, -25*25)
        input_hist_flat[0, 25*25*3::] = rgb2gray(state.pob).flatten()
        input_hist = input_hist_flat.reshape(1, 4, 25, 25)
        action = np.argmax(agent.predict(input_hist, batch_size=32, verbose=0))
        state = sim.step(action)
        """
        action = randrange(opt.act_num)
        state = sim.step(action)
        print('action shape:', action)
        print('state shape:', state)
        """
        epi_step += 1

    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

    if step % opt.prog_freq == 0:
        print(step)

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

# 2. calculate statistics
print(float(nepisodes_solved) / float(nepisodes))
# 3. TODO perhaps  do some additional analysis
