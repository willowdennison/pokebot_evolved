#the definition for the NameModel class to be used in the trainer.py file
import torch
import torch.nn as nn

class NameModel(nn.Module): #uses the Module class as parent
    
    def __init__(self, vocab_length, input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout_prob=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, vocab_length)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] #only use the last output
        x = self.linear(self.dropout(x)) #produce output
        return x