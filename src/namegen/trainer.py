# trains a character-by-character text generation model to generate custom pokemon names using pytorch

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from namegen.nameModel import NameModel

names = 'namegen/data/names.csv'

raw_text = open(names, encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text))) #list of used characters
char_to_int = dict((c, i) for i, c in enumerate(chars)) #mapping chars to int

print(chars) 

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_len = 20 #sequence length

X_data = []
y_data = []

for i in range(0, n_chars - seq_len):
    seq_in = raw_text[i : i + seq_len] #input: slices from dataset of length 5
    seq_out = raw_text[i + seq_len] #the character following the input
    
    X_data.append([char_to_int[char] for char in seq_in]) #encode characters as ascii values
    y_data.append(char_to_int[seq_out])

#X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1)

n_patterns = len(X_data) #X_train
print("Total Patterns: ", n_patterns)

X = torch.tensor(X_data, dtype=torch.float32).reshape(n_patterns, seq_len, 1) #X_train
X = X / float(n_vocab) #normalize ascii values between 0 and 1 - helps model with no information loss

y = torch.tensor(y_data) #y_train

# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(len(X_test), seq_len, 1)
# X_test_tensor = X_test_tensor / float(n_vocab)
# y_test_tensor = torch.tensor(y_test)

print(X.shape, y.shape)

n_epochs = 500
batch_size = 128
model = NameModel(vocab_length=n_vocab, num_layers=2) #intialize NameModel with default parameters - I wrote the class myself, default values are tuned to this project
#layers=2 may make this run horribly slow, just testing right now

optimizer = optim.Adam(model.parameters())
loss_f = torch.nn.CrossEntropyLoss(reduction="sum") #loss function
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size) #shuffles data into batches for training

# test_loader = loader = data.DataLoader(data.TensorDataset(X_test_tensor, y_test_tensor), shuffle=True, batch_size=batch_size)


best_model = None
best_loss = np.inf

for epoch in range(n_epochs):
    
    model.train()
    
    for X_batch, y_batch in loader:
        
        y_pred = model(X_batch)
        
        loss = loss_f(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval() #set to evaluation mode
        loss = 0 #reset to 0 to total loss across all batches
        
        with torch.no_grad(): #reduces memory use
            #for X_batch, y_batch in loader:
            for X_batch, y_batch in loader:
                y_pred = model(X_batch) #X_batch
                loss += loss_f(y_pred, y_batch) #y_batch
            
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
                
            print("Epoch %d: Cross-entropy: %.4f" % (epoch + 1, loss))
    
    path = f"namegen/weights/12-11 2 hidden layers/namegenweightsepoch{epoch+1}loss{int(loss)}.pth"
    torch.save([best_model, char_to_int, n_vocab, loss], path)
    print(f'\nModel weights saved as {path}\n')

torch.save([best_model, char_to_int, n_vocab, loss], path)
print(f'\nModel weights saved as {path}\n')
print('Training complete.')