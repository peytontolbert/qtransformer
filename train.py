# train.py

import torch
from q_transformer import QTransformer
from agent import Agent
from text_environment import TextEnvironment
from replay_memory import ReplayMemory

def main():
    num_episodes = 10
    batch_size = 2

    model = QTransformer()
    env = TextEnvironment()
    memory = ReplayMemory(10000)
    agent = Agent(model, env, memory)

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()

        for t in range(100):  # Max steps per episode
            # Select and perform an action
            action = agent.select_action([state])
            next_state, reward, done = env.step(action.item())

            # Store the transition in memory
            agent.memory.push([state], action, [next_state] if next_state else None, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            agent.optimize_model(batch_size)

            if done:
                break

    # Save the trained model
    torch.save(model.state_dict(), 'q_transformer_model.pth')

if __name__ == '__main__':
    main()
