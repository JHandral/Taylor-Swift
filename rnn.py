#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# In[2]:


all_chars = string.printable
n_chars = len(all_chars)
file = open('taylor_swift_lyrics_clean.txt', 'r').read()
file_len = len(file)
print('Length of file: {}'.format(file_len))
print('All possible characters: {}'.format(all_chars))
print('Number of all possible characters: {}'.format(n_chars))


# In[3]:


# Get a random sequence of the dataset.
def get_random_seq():
    seq_len = 128 # The length of an input sequence.
    start_index = random.randint(0, file_len - seq_len)
    end_index = start_index + seq_len + 1
    return file[start_index:end_index]
# Convert the sequence to one-hot tensor.

def seq_to_onehot(seq):
    tensor = torch.zeros(len(seq), 1, n_chars)
    for t, char in enumerate(seq):
        if char not in all_chars:
            continue  # skip this character
        index = all_chars.index(char)
        tensor[t][0][index] = 1
    return tensor

def seq_to_index(seq):
    tensor = torch.zeros(len(seq), 1)
    for t, char in enumerate(seq):
        if char not in all_chars:
            continue  # skip this character
        tensor[t] = all_chars.index(char)
    return tensor


# Sample a mini-batch including input tensor and target tensor.
def get_input_and_target():
    seq = get_random_seq()
    input = seq_to_onehot(seq[:-1]) # Input is represented in one-hot.
    target = seq_to_index(seq[1:]).long() # Target is represented in index.
    return input, target




# In[4]:


# If there are GPUs, choose the first one for computing. Otherwise use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# If 'cuda:0' is printed, it means GPU is available.


# In[5]:


import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        # Initialization.
        super(Net, self).__init__()
        self.input_size = n_chars # Input size: Number of unique chars.
        self.hidden_size = 150 # Hidden size: 100.
        self.output_size = n_chars # Output size: Number of unique chars.
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        # Forward function.
        hidden = self.rnn(input, hidden)
        output = self.linear(hidden)
        return output, hidden

    def init_hidden(self):
        # Initial hidden state.
        # 1 means batch size = 1.
        return torch.zeros(1, self.hidden_size).to(device)

net = Net() # Create the network instance.
net.to(device) # Move the network parameters to the specified device


# In[6]:


# Training step function.
def train_step(net, opt, input, target):
    """ Training step.
    net: The network instance.
    opt: The optimizer instance.
    input: Input tensor. Shape: [seq_len, 1, n_chars].
    target: Target tensor. Shape: [seq_len, 1].
    """
    seq_len = input.shape[0] # Get the sequence length of current input.
    hidden = net.init_hidden() # Initial hidden state.
    net.zero_grad() # Clear the gradient.
    loss = 0 # Initial loss.
    
    for t in range(seq_len): # For each one in the input sequence.
        output, hidden = net(input[t], hidden)
        loss += loss_func(output, target[t])
        
    
    loss.backward() # Backward.
    opt.step() # Update the weights.
    return loss / seq_len # Return the average loss w.r.t sequence length.


# In[7]:


# Evaluation step function.
def eval_step(net, init_seq='W', predicted_len=100):
    # Initialize the hidden state, input and the predicted sequence.
    hidden = net.init_hidden()
    init_input = seq_to_onehot(init_seq).to(device)
    predicted_seq = init_seq

    # Use initial string to "build up" hidden state.
    for t in range(len(init_seq) - 1):
        output, hidden = net(init_input[t], hidden)

    # Set current input as the last character of the initial string.
    input = init_input[-1]
    # Predict more characters after the initial string.
    for t in range(predicted_len):
        # Get the current output and hidden state.
        output, hidden = net(input, hidden)
        # Sample from the output as a multinomial distribution.
        predicted_index = torch.multinomial(output.view(-1).exp(), 1)[0]
        # Add predicted character to the sequence and use it as next input.
        predicted_char = all_chars[predicted_index]
        predicted_seq += predicted_char
        # Use the predicted character to generate the input of the next round.
        input = seq_to_onehot(predicted_char)[0].to(device)
    return predicted_seq


# In[ ]:


# Number of iterations.
iters = 30000 # Number of training iterations.
print_iters = 100 # Number of iterations for each log printing.
# The loss variables.
all_losses = []
loss_sum = 0
#iters_checkpoints = [5000, 10000, 15000, 20000, 25000, 30000]
#generated_texts = []

# Initialize the optimizer and the loss function.
opt = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()

# Training procedure.
for i in range(iters):
    input, target = get_input_and_target() # Fetch input and target.
    input, target = input.to(device), target.to(device) # Move to GPU memory.
    loss = train_step(net, opt, input, target) # Calculate the loss.
    loss_sum += loss # Accumulate the loss.

    # Print the log.
    if i % print_iters == print_iters - 1:
        all_losses.append(loss_sum / print_iters) 
        print('iter:{}/{} loss:{}'.format(i, iters, loss_sum / print_iters))
        print('generated sequence: {}\n'.format(eval_step(net))) 
        loss_sum = 0 # reset the sum

    


# In[ ]:


plt.xlabel('iters')
plt.ylabel('loss')
plt.plot(all_losses)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




