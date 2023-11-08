

import gym
import numpy as np
from PIL import Image
import torch
from breakout import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = DQNBreakout(device=device, render_mode='human')

state = environment.reset()

for _ in range(1000):
    environment.render()
    action = environment.action_space.sample()
    state, reward, done, info = environment.step(action)
    if done:
        environment.reset()
