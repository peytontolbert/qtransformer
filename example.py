# run_model.py

import torch
from q_transformer import QTransformer

def main():
    # Load the trained model
    model = QTransformer()
    model.load_state_dict(torch.load('q_transformer_model.pth'))
    model.eval()

    # Get user input
    user_input = input("User: ")

    # Get Q-values for actions
    with torch.no_grad():
        q_values = model([user_input])
        action = q_values.argmax(dim=1).item()

    actions = ['respond', 'code_execute']
    selected_action = actions[action]
    print(f"Agent selected action: {selected_action}")

    # Generate text based on action
    generated_text = model.generate_text(user_input, action)
    print(f"Agent output: {generated_text[0]}")

if __name__ == '__main__':
    main()
