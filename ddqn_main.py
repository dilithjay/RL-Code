import gym
import numpy as np
from dueling import DuelingDQNAgent
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('CartPole-v0')

# Get the state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Set the random seed
seed = 0

# Create the DDQN agent
agent = DuelingDQNAgent(state_size, action_size, seed)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 1000
max_steps = 1000

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# Set the rewards and scores lists
rewards = []
scores = []

# Run the training loop
for i_episode in range(num_episodes):
    print(f'Episode: {i_episode}')
    # Initialize the environment and the state
    state = env.reset()[0]
    score = 0
    # eps = eps_end + (eps_start - eps_end) * np.exp(-i_episode / eps_decay)
    # Update the exploration rate
    eps = max(eps_end, eps_decay * eps)
    
    # Run the episode
    for t in range(max_steps):
        # Select an action and take a step in the environment
        action = agent.act(state, eps)
        next_state, reward, done, _, _ = env.step(action)
        # Store the experience in the replay buffer and learn from it
        agent.step(state, action, reward, next_state, done)
        # Update the state and the score
        state = next_state
        score += reward
        # Break the loop if the episode is done
        if done:
            break
        
    print(f"\tScore: {score}, Epsilon: {eps}")
    # Save the rewards and scores
    rewards.append(score)
    scores.append(np.mean(rewards[-100:]))

# Close the environment
env.close()

plt.ylabel("Score")
plt.xlabel("Episode")
plt.plot(range(len(rewards)), rewards)
plt.plot(range(len(rewards)), scores)
plt.legend(['Reward', "Score"])
plt.show()
