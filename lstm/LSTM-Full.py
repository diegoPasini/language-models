import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import time

BATCH_SIZE = 32
CONTEXT_SIZE = 96

# First Read the dataset:
file = open("/language-models/tiny-shakespeare.txt", "r")
contents = file.read()
file.close()

vocabulary = list(set(contents))
vocabulary = sorted(vocabulary)
VOCAB_SIZE = len(vocabulary)

print("Vocabulary Length: ", len(vocabulary))
print("Content Length: ", len(contents))

assert torch.cuda.is_available()


## First we must make encode and decoder functions for our dataset
string_to_int = {ch : i for i, ch in enumerate(vocabulary)}
int_to_string = {i : ch for i, ch in enumerate(vocabulary)}
encode = lambda s : [string_to_int[c] for c in s]
decode = lambda l : ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(contents), dtype=int, device = 'cuda')

n = int(0.9*len(data))
train_data = data[n:]
val_data = data[:n]
train_data = train_data.float()
val_data = val_data.float()

## Now, we define our context window
torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in ix])
    return x, y

## LSTM Model
## Short Term Memory Block
class stmBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Forget Gate:
        self.Wif = nn.Linear(input_dim, hidden_dim, bias = False)
        self.Whf = nn.Linear(hidden_dim, hidden_dim)

        # Input Gate:
        self.Wii = nn.Linear(input_dim, hidden_dim, bias = False)
        self.Whi = nn.Linear(hidden_dim, hidden_dim)

        # Candidate Gate:
        self.Wic = nn.Linear(input_dim, hidden_dim, bias = False)
        self.Whc = nn.Linear(hidden_dim, hidden_dim)

        # Output Gate:
        self.Wio = nn.Linear(input_dim, hidden_dim, bias = False)
        self.Who = nn.Linear(hidden_dim, hidden_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm((BATCH_SIZE, input_dim, hidden_dim))

    def forward(self, x, c_prev = None, h_prev = None):
        # If not first block:'
        residual_x = x
        if h_prev is not None:
            f = self.sigmoid(self.dropout(self.Wif(x)) + self.dropout(self.Whf(h_prev)))
            i = self.sigmoid(self.dropout(self.Wii(x)) + self.dropout(self.Whi(h_prev)))
            c = self.tanh(self.dropout(self.Wic(x)) + self.dropout(self.Whc(h_prev)))
            o = self.sigmoid(self.dropout(self.Wio(x)) + self.dropout(self.Who(h_prev)))
        else:
            f = self.sigmoid(self.dropout(self.Wif(x)))
            i = self.sigmoid(self.dropout(self.Wii(x)))
            c = self.tanh(self.dropout(self.Wic(x)))
            o = self.sigmoid(self.dropout(self.Wio(x)))

        if c_prev == None:
            c_t = i * c
        else:
            c_t = f * c_prev + i * c
        h = o * self.tanh(c_t)
        return c_t, h

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.stmBlocks = nn.ModuleList([stmBlock(input_dim, hidden_dim) for i in range(CONTEXT_SIZE)])
        self.linLays = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(CONTEXT_SIZE)])
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, targets=None):
        logits = []
        h = None
        c = None
        for i in range(CONTEXT_SIZE): 
            element = x[:, i].unsqueeze(1)
            c, h = self.stmBlocks[i](element, c, h) if h is not None else self.stmBlocks[i](element)
            o = self.dropout(self.linLays[i](h))
            logits.append(o)
        
        logits = torch.stack(logits, dim=1)
        if targets is not None:
            loss = self.loss(logits.view(-1, VOCAB_SIZE), targets.view(-1).long())
            return logits, loss
        return logits, None


def evaluate(model):
    lossAvg = 0
    counter = 0
    for i in range(400):
        counter+=1
        batch = get_batch("test")
        logits, loss = model(batch[0], batch[1])
        lossAvg += loss.item()
    return lossAvg / counter


model = LSTM(1, 512, VOCAB_SIZE)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
total_params = sum(p.numel() for p in model.parameters())
print(f"------Number of parameters: {total_params}--------")


# Code here

# Calculate the end time and time taken
start = time.time()
training_iterations = 10000
for j in range(training_iterations // 100):
    for i in tqdm(range(100)):
        optimizer.zero_grad()
        batch = get_batch("train")
        outputs, loss = model(batch[0], batch[1])
        loss.backward()
        #print("Loss: ", loss.item())
        optimizer.step()
    print(f"Training Iteration: {i}, Loss: {loss}")

end = time.time()
length = end - start
print("Traing Time: ", length)
print("Loss: ", loss.item())

print("Test Loss: ", evaluate(model))

## Let's do some testing:
def infer(input_data):
    with torch.no_grad():
        outputs, _ = model(input_data)
        predicted_indices = torch.argmax(outputs, dim=-1)
        return predicted_indices
        
test_batch = get_batch("test")
result = infer(test_batch[0])
for i in range(BATCH_SIZE):
    print("Input: ", decode(test_batch[0].tolist()[i]))
    print(decode(result.tolist()[i]))
