# text_environment.py

import torch

class TextEnvironment:
    def __init__(self):
        self.state = None
        self.gamma = 0.99

    def reset(self):
        # Return initial state (e.g., user instruction)
        self.state = input("User: ")
        return self.state

    def step(self, action):
        # Simulate environment response to action
        if action == 0:
            # 'respond' action
            print("Agent is responding to the user...")
            reward = torch.tensor([1.0])  # Placeholder reward
            done = True
            next_state = None
        elif action == 1:
            # 'code_execute' action
            print("Agent is executing code...")
            reward = torch.tensor([1.0])  # Placeholder reward
            done = True
            next_state = None
        else:
            reward = torch.tensor([0.0])
            done = True
            next_state = None

        return next_state, reward, done
