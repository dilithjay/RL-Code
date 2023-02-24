import gym  # import the OpenAI Gym library
import numpy as np  # import NumPy library

# Create the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Set parameters
alpha = 0.1  # learning rate
gamma = 0.6  # discount factor

# Set the action and state space sizes
n_actions = env.action_space.n
n_states = env.observation_space.n

# Initialize Q-table with zeros
Q = np.zeros((n_states, n_actions))

# Set number of episodes
num_episodes = 10000

# Create a list to store rewards
r_list = []

# Loop through episodes
for i in range(num_episodes):
    # Reset the environment and get the initial state
    state = env.reset()[0]
    r_all = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state,:] + np.random.randn(1, n_actions) / (i+1))
        
        # Get new state and reward from environment
        new_state, reward, done, _, _ = env.step(action)

        # Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        
        # Update total reward
        r_all += reward
        
        # Set new state
        state = new_state

    # Append total reward for this episode to the reward list
    r_list.append(r_all)

print("Score over time: " +  str(sum(r_list)/num_episodes))
print("Final Q-Table Values")
print(Q)


# ---------- Simulate policy ----------
policy = {}
for i, act in enumerate(np.argmax(Q, axis=1)):
    policy[i] = act
    
print("Policy:", policy)

path = []
dirs = ['left', 'down', 'right', 'up']

state = env.reset()[0]
r_all = 0
done = False
while not done:
    # Choose an action by greedily (with noise) picking from Q table
    action = np.argmax(Q[state,:])
    path.append(dirs[action])

    # Get new state and reward from environment
    new_state, reward, done, _, _ = env.step(action)
    
    # Update total reward
    r_all += reward
    
    # Set new state
    state = new_state

print("Path:", path)
print("Reward:", reward)
