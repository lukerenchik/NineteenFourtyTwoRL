from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from PIL import Image, ImageDraw, ImageFont
import os
import random
from collections import deque


actions = ['', 'a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']


class NFT_Environment(gym.Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self.debug = debug
        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 160, 144), dtype=np.uint8)

        self._fitness = 0
        self._previous_fitness = 0
        self.ticks_survived = 0
        self.previous_observation = None
        self.static_frame_count = 0
        self.frame_history = deque(maxlen=50)
        self.static_frame_penalty = 0

        # Set up a directory for saving samples
        #self.sample_dir = "samples"
        #os.makedirs(self.sample_dir, exist_ok=True)
        #self.sample_count = 0  # How many samples have been saved
        #self.max_samples = 50  # Max samples to save

        self.start_stage_one()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Execute the action
        if action != 0:
            self.pyboy.button(actions[action])
        self.pyboy.tick(1)

        # Process the current frame
        observation = self._process_game_area()
        #self.frame_history.append(observation)

        # Compare current frame to 10 frames ago
        #if len(self.frame_history) == 50:
        #    if np.array_equal(observation, self.frame_history[0]):
        #        self.static_frame_penalty = self.static_frame_penalty + 1  # Cap at 10
        #    else:
        #        self.static_frame_penalty = 0  # Reset penalty if screen changes

        # Static penalty applied
        #static_penalty = -1 * self.static_frame_penalty

        # Check done condition
        if self.pyboy.memory[49820] == 255:
            done = True
            self.ticks_survived = 0
        else:
            done = False
            self.ticks_survived += 1

        # Fitness and rewards
        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness 

        print(f"Reward: {reward}, Static Penalty: {0}, ")

        info = {}
        truncated = False

        return observation, reward, done, truncated, info


    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        current_score = self.get_score()
        current_health = self.get_lives()
        ships_destroyed = self.get_enemies_destoryed()
        current_dodges = self.get_dodges()


        self._fitness += ships_destroyed * 100
        self._fitness += current_health * 20
        self._fitness += current_dodges * 5
        self._fitness -= 0.01 * self.ticks_survived

    def reset(self, **kwargs):
        self.start_stage_one()
        self._fitness = 0
        self._previous_fitness = 0
        observation = self._process_game_area()
        info = {}
        return observation, info

    def start_stage_one(self):
        with open("GBC/start_of_game.state", "rb") as f:
            self.pyboy.load_state(f)

    def get_lives(self):
        return self.pyboy.memory[49820]

    def get_score(self):
        return self.pyboy.memory[49825]

    def get_dodges(self):
        return self.pyboy.memory[49822]

    def get_enemies_destoryed(self):
        return self.pyboy.memory[49815]

    def close(self):
        self.pyboy.stop()


    def _save_sample(self, observation):
        # observation is the game_area (e.g. 32x32)
        # Save the array to a .npy file
        array_path = os.path.join(self.sample_dir, f"game_area_{self.sample_count}.npy")
        np.save(array_path, observation)

        # Get a screenshot (PIL Image)
        screenshot = self.pyboy.screen.image
        screenshot = screenshot.convert("RGBA")

        # Create a semi-transparent white overlay (30% opacity)
        overlay = Image.new("RGBA", screenshot.size, (255, 255, 255, int(255 * 0.3)))
        base_image = Image.alpha_composite(screenshot, overlay)

        # ----- New scaling approach -----
        # Let's scale the base image up to give more room for text
        scale_factor = 4  # Increase if needed for more space
        orig_width, orig_height = base_image.size
        new_width = orig_width * scale_factor
        new_height = orig_height * scale_factor
        # Use the new Pillow Resampling interface
        base_image = base_image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

        # Now recalculate cell sizes based on the scaled image
        obs_height, obs_width = observation.shape
        cell_width = new_width / obs_width
        cell_height = new_height / obs_height

        draw = ImageDraw.Draw(base_image)
        font = ImageFont.load_default()

        # Draw each cell value at a scaled position
        for y in range(obs_height):
            for x in range(obs_width):
                val = observation[y, x]
                text_str = f"{val:3d}"

                # Position text in the center of each cell
                text_x = x * cell_width + cell_width / 2
                text_y = y * cell_height + cell_height / 2

                # Get text bounding box to center the text
                bbox = draw.textbbox((0, 0), text_str, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                centered_x = text_x - text_w / 2
                centered_y = text_y - text_h / 2

                draw.text((centered_x, centered_y), text_str, font=font, fill=(0, 0, 0, 255))

        # Save the composed image
        image_path = os.path.join(self.sample_dir, f"screenshot_{self.sample_count}.png")
        base_image.save(image_path)

        self.sample_count += 1


    def _process_game_area(self):
        screenshot = self.pyboy.screen.image
        grayscale = screenshot.convert("L")
        resized = grayscale.resize((144,160), Image.Resampling.BILINEAR)
        observation = np.array(resized, dtype=np.uint8)
        observation = np.expand_dims(observation, axis=0)
        return observation
