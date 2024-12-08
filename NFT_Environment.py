from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy



actions = ['', 'a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

gbc_gamespace_shape = (32, 32)

NFT_observation_space = spaces.Box(low=0, high=500, shape=gbc_gamespace_shape, dtype=np.uint16)

class NFT_Environment(gym.Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy

        #These correspond to rewards, but may be unneccesary for systems with external agents.
        self._fitness = 0
        self._previous_fitness = 0


        self.debug = debug

        if not self.debug:
            #Setting emulation speed to 0 allows for fastest possible execution
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = NFT_observation_space
        self.start_stage_one()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if action == 0:
            pass
        else:
            self.pyboy.button(actions[action])
        self.pyboy.tick(1)

        if self.pyboy.memory[49820] == 255:
            done = 1
        else:
            done = 0


        self._calculate_fitness()

        reward = self._fitness-self._previous_fitness

        observation = self.pyboy.game_area()
        
        info = {}

        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        current_score = self.get_score()
        current_health = self.get_lives()
        ships_destroyed = self.get_enemies_destoryed()
        current_dodges = self.get_dodges()
        self._fitness = 0
        #self._fitness += current_score
        self._fitness += ships_destroyed * 10
        self._fitness += current_health * 150
        self._fitness += current_dodges * 1000


    def reset(self, **kwargs):
        self.start_stage_one()
        self._fitness = 0
        self._previous_fitness = 0
        observation = self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def start_stage_one(self):
        with open("start_of_game.state", "rb") as f:
            self.pyboy.load_state(f)

    def get_lives(self):
        health = self.pyboy.memory[49820]
        return health

    def get_score(self):
        score = self.pyboy.memory[49825]
        return score

    def get_dodges(self):
        dodges = self.pyboy.memory[49822]
        return dodges

    def get_enemies_destoryed(self):
        enemies_destroyed = self.pyboy.memory[49815]
        return enemies_destroyed
    
    def get_game_area(self):
        current_map = self.pyboy.game_area()
        return current_map

    def close(self):
        self.pyboy.stop()

    