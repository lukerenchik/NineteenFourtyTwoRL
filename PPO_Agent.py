from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from NFT_Environment import NFT_Environment
from pyboy import PyBoy
import atexit


def create_env():
    #pyboy = PyBoy("GBC/nineteenFT.gbc", window='null')
    pyboy = PyBoy("GBC/nineteenFT.gbc")
    return NFT_Environment(pyboy)

n_env = 1
env = make_vec_env(create_env, n_envs=n_env)

model_path = "ppo_nft_agent.zip"

try:
    model = PPO.load(model_path, env=env)
    print("Loaded Existing Model.")
except FileNotFoundError:
    print("No saved model found. Creating a new model.")
    model = PPO("CnnPolicy", env, verbose=1)

def save_model():
    print("Saving model...")
    model.save(model_path)
    print("Model saved successfully")

atexit.register(save_model)

try:
    print("Starting Training...")
    model.learn(total_timesteps=50000)
except KeyboardInterrupt:
    print("Training Interrupted Manually")
