import math
import copy
import random
import numpy as np
import gymnasium as gym


# Define the node class
class Node:
    def __init__(self, state=None):
        self.value = 0
        self.n = 0
        self.state = state
        self.children = []

    # Method to get the child with the max UCB1 score and it's index
    def get_max_ucb1_child(self):
        if not self.children:
            return None

        max_i = 0
        max_ucb1 = float("-inf")

        for i, child in enumerate(self.children):
            ucb1 = self.get_child_ucb1(child)

            if ucb1 > max_ucb1:
                max_ucb1 = ucb1
                max_i = i

        return self.children[max_i], max_i

    # Method to calculate the UCB1 score of a child
    def get_child_ucb1(self, child):
        if child.n == 0:
            return float("inf")
        return child.value / child.n + 2 * math.sqrt(math.log(self.n, math.e) / child.n)


class MCTS:
    def __init__(self, env, reset=False):
        self.env = env
        if reset:
            start_state, _ = self.env.reset()
        else:
            start_state = self.env.unwrapped.state
        self.start_env = copy.deepcopy(self.env)
        self.root_node = Node(start_state)

        for act in range(self.env.action_space.n):
            env_copy = copy.deepcopy(self.env)
            new_state, _, _, _, _ = env_copy.step(act)
            new_node = Node(new_state)
            self.root_node.children.append(new_node)

    # Run `n_iter` number of iterations
    def run(self, n_iter=200):
        for _ in range(n_iter):
            value, node_path = self.traverse()
            self.backpropagate(node_path, value)
            self.env = copy.deepcopy(self.start_env)

        vals = [float("-inf")] * self.env.action_space.n
        for i, child in enumerate(self.root_node.children):
            vals[i] = (child.value / child.n) if child.n else 0

        return np.exp(vals) / sum(np.exp(vals))

    def traverse(self):
        cur_node = self.root_node
        node_path = [cur_node]
        while cur_node.children:
            cur_node, idx = cur_node.get_max_ucb1_child()
            self.env.step(idx)
            node_path.append(cur_node)

        if cur_node.n:
            for act in range(self.env.action_space.n):
                env_copy = copy.deepcopy(self.env)
                new_state, _, _, _, _ = env_copy.step(act)
                new_node = Node(new_state)
                cur_node.children.append(new_node)
                
            cur_node, idx = cur_node.get_max_ucb1_child()
            self.env.step(idx)
            node_path.append(cur_node)
        
        tot_reward = self.rollout()

        return tot_reward, node_path

    def backpropagate(self, node_path: list, last_value: float):
        for node in node_path[::-1]:
            node.value += last_value
            node.n += 1

    def rollout(self) -> float:
        tot_reward = 0

        while True:
            act = random.randrange(self.env.action_space.n)
            _, reward, done, _, _ = self.env.step(act)
            tot_reward += reward

            if done:
                break

        return tot_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    
    done = False
    tot_reward = 0
    
    while not done:
        mcts = MCTS(copy.deepcopy(env), reset=False)
        probs = mcts.run(20)
        action = random.choices(range(len(probs)), weights=probs, k=1)[0]
        
        _, reward, done, _, _ = env.step(action)
        tot_reward += reward
        print(f"Return: {tot_reward}   ", end='\r')
