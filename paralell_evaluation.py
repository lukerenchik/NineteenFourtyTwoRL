from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from NFT_Environment import NFT_Environment
from pyboy import PyBoy

# Create the environment function
def create_env():
    pyboy = PyBoy("GBC/nineteenFT.gbc")  # Headless for efficiency
    return NFT_Environment(pyboy)

# Load vectorized environments (4 environments)
n_envs = 4
env = make_vec_env(create_env, n_envs=n_envs)

# Load the trained model
model_path = "ppo_nft_agent.zip"
model = PPO.load(model_path, env=env)
print("Loaded trained model for parallel evaluation.")

# Evaluation loop for vectorized environments
def evaluate_model_parallel(env, model, n_episodes=10):
    total_rewards = [0 for _ in range(n_envs)]  # Rewards for all environments
    episode_counts = [0 for _ in range(n_envs)]
    episodes_completed = 0

    obs = env.reset()
    while episodes_completed < n_episodes:
        actions, _ = model.predict(obs)  # Predict actions for all environments
        obs, rewards, dones, infos = env.step(actions)  # Step through all environments

        # Track rewards
        for i in range(n_envs):
            total_rewards[i] += rewards[i]
            if dones[i]:  # If an episode finishes
                print(f"Env {i + 1}: Episode {episode_counts[i] + 1} Total Reward = {total_rewards[i]}")
                total_rewards[i] = 0  # Reset reward for the next episode
                episode_counts[i] += 1
                episodes_completed += 1

    avg_episodes_completed = sum(episode_counts) / n_envs
    print(f"Completed {n_episodes} episodes across {n_envs} environments in parallel.")
    print(f"Average Episodes Per Environment: {avg_episodes_completed}")

# Run the parallel evaluation
evaluate_model_parallel(env, model, n_episodes=40)

# Clean up
env.close()
