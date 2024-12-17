from stable_baselines3 import PPO
from NFT_Environment import NFT_Environment
from pyboy import PyBoy

def create_env():
    pyboy = PyBoy("GBC/nineteenFT.gbc")  # Run headless for evaluation
    return NFT_Environment(pyboy)

# Load environment
env = create_env()

# Load the trained model
model_path = "ppo_nft_agent.zip"
model = PPO.load(model_path, env=env)
print("Loaded trained model for evaluation.")

# Evaluation loop
def evaluate_model(env, model, n_episodes=10):
    total_rewards = []

    for episode in range(n_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            # Use the trained model to predict actions
            action, _ = model.predict(observation)
            observation, reward, done, truncated, info = env.step(action)

            # Accumulate rewards
            episode_reward += reward
            step_count += 1

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Steps = {step_count}")
        total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / n_episodes
    print(f"Average Reward over {n_episodes} episodes: {avg_reward}")

# Run the evaluation
evaluate_model(env, model, n_episodes=5)

# Clean up
env.close()
