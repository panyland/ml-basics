import gymnasium as gym
import numpy as np

# Create environment and render
env = gym.make("Taxi-v3",render_mode='ansi')
env.reset()
print(env.render())

# Number of states and possible actions in the env
STATES = env.observation_space.n
ACTIONS = env.action_space.n

# Initialize Q-table
Q_table = np.zeros((STATES, ACTIONS))

# Determine number of episodes, maximum actions per episode and constants for updating Q-table
episodes = 1000
max_actions = 50
alfa = 0.81
gamma = 0.96

# Probability of taking a random action
random_chance = 0.9

# Initialize arrays for number of actions and total reward per episodes
actions = []
rewards = []

# Run for number of episodes
for episode in range(episodes):

    # Initial state when new episode
    state, _ = env.reset()

    # Run for max number of actions per episode
    for _ in range(max_actions):
        # Variable to count number of actions in this episode
        num_actions = 0
        # Determine whether random action or max from Q-table
        if np.random.uniform(0, 1) < random_chance:
            action = env.action_space.sample()
            num_actions += 1
        else: 
            action = np.argmax(Q_table[state, :])
            num_actions += 1

        # Information from taking certain action
        next_state, reward, done, _ , _ = env.step(action)
        # Updating Q-table based on acquired info
        Q_table[state, action] = Q_table[state, action] + alfa*(reward + gamma*np.max(Q_table[next_state, :]) -  Q_table[state, action])
        # Set new state as current
        state = next_state

        # End if task completed and append total reward and num of actions
        if done: 
            rewards.append(reward)
            actions.append(num_actions)
            break

# Print results
print(Q_table)
print(f"Average total reward: {sum(rewards)/len(rewards)}")
print(f"Average number of actions: {sum(actions)/len(actions)}")
