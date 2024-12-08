from stable_baselines3 import PPO
from NFT_Environment import NFT_Environment
from pyboy import PyBoy

pyboy = PyBoy("nineteenFT.gbc")


nft_env = NFT_Environment(pyboy)



model = PPO("MlpPolicy", nft_env, verbose=1)
model.learn(total_timesteps=100000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
 

for i in range(100000):
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)


for i in range(10000):
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    #vec_env.render()
    model.save("first_NFT")

nft_env.close()