# q_transformer.py

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class QTransformer(nn.Module):
    def __init__(self, num_actions=2, hidden_dim=768):
        super(QTransformer, self).__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Pre-trained tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        # Q-Network head
        self.q_head = nn.Linear(hidden_dim, num_actions)

        # Text generation decoder (optional)
        self.text_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, self.text_encoder.config.vocab_size)

    def forward(self, input_text):
        # Tokenize and encode input text
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.text_encoder(**inputs)

        # Use [CLS] token representation for Q-values
        cls_representation = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
        q_values = self.q_head(cls_representation)               # Shape: [batch_size, num_actions]

        return q_values

    def generate_text(self, input_text, action, max_length=50):
        # Tokenize and encode input text
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        encoder_outputs = self.text_encoder(**inputs)

        # Prepare decoder input (start with [CLS] token representation)
        decoder_input = encoder_outputs.last_hidden_state[:, 0, :].unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]

        generated_tokens = []

        hidden = None  # Initial hidden state for GRU

        for _ in range(max_length):
            output, hidden = self.text_decoder(decoder_input, hidden)
            token_logits = self.output_head(output)  # Shape: [batch_size, 1, vocab_size]
            next_token = torch.argmax(token_logits, dim=-1)  # Greedy decoding
            generated_tokens.append(next_token)

            # Prepare next decoder input
            decoder_input = output

        # Convert token IDs to words
        generated_token_ids = torch.cat(generated_tokens, dim=1)  # Shape: [batch_size, seq_length]
        generated_text = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        return generated_text
