from stable_baselines3 import PPO
from NFT_Environment import NFT_Environment

nft_env = NFT_Environment()

model = PPO("nftPolicy", nft_env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

nft_env.close()