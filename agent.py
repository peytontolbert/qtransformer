# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from q_transformer import QTransformer
from text_environment import TextEnvironment
from replay_memory import ReplayMemory, Transition

class Agent:
    def __init__(self, model, environment, replay_memory, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500):
        self.model = model
        self.env = environment
        self.memory = replay_memory
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        sample = torch.rand(1).item()
        epsilon_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                            torch.exp(torch.tensor(-1. * self.steps_done / self.epsilon_decay))
        self.steps_done += 1

        if sample > epsilon_threshold:
            with torch.no_grad():
                q_values = self.model(state)
                action = q_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[torch.randint(0, self.model.num_actions, (1,)).item()]], dtype=torch.long)

        return action

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = batch.next_state

        # Compute Q(s_t, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1})
        next_state_values = torch.zeros(batch_size)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), dtype=torch.bool)
        non_final_next_states = [s for s in next_state_batch if s is not None]

        if non_final_next_states:
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.env.gamma) + reward_batch

        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
