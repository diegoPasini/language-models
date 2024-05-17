import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

# First Read the dataset:
file = open("/Users/diego/Scripts/og-language-models/tiny-cicero.txt", "r")
contents = file.read()
#print(contents)
file.close()


vocabulary = list(set(contents))
vocabulary = sorted(vocabulary)
VOCAB_SIZE = len(vocabulary)
print(vocabulary)


print("Vocabulary Length: ", len(vocabulary))
print("Content Length: ", len(contents))

## First we must make encode and decoder functions for our dataset
string_to_int = {ch : i for i, ch in enumerate(vocabulary)}
int_to_string = {i : ch for i, ch in enumerate(vocabulary)}
encode = lambda s : [string_to_int[c] for c in s]
decode = lambda l : ''.join([int_to_string[i] for i in l])
print(encode("Hello World"))
print(decode(encode("Hello World")))


data = torch.tensor(encode(contents), dtype=int)
print(data.shape,data.dtype)
print(data[:1000])


n = int(0.9*len(data))
train_data = data[n:]
val_data = data[:n]
train_data = train_data.float()
val_data = val_data.float()


## Now, we define our context window
torch.manual_seed(1337)
BATCH_SIZE = 4
CONTEXT_SIZE = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb)
print('targets:')
print(yb)


## Now, for the RNN

## First we make the first step layer
class step(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(1)
        self.w_xh = nn.Linear(1, 256 , bias = False)
        self.norm2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.2)
        self.w_hy = nn.Linear(256, VOCAB_SIZE, bias = False)
        #self.norm3 = nn.BatchNorm1d(256)
        self.w_hh = nn.Linear(256, 256, bias = False)
    
    def forward(self, x, h_i = None, ):
        x = self.norm1(x)
        h_1 = self.w_xh(x)
        if (h_i != None):
            h_1 = h_1 + self.w_hh(h_i)
        h_1 = self.dropout(h_1)
        h_1 = self.norm2(h_1)
        y = self.w_hy(h_1)
        return y, h_1 

class RNN(nn.Module):
    def __init__(self, context_size):
        super().__init__()
        self.rnnSize = context_size
        self.steps = nn.ModuleList([step() for i in range(context_size)])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        assert x.shape[1] == self.rnnSize, "Input size must be equal to context size."
        logits = []
        h = None
        for i in range(self.rnnSize):
            element = x[:, i].unsqueeze(1)
            y, h = self.steps[i](element, h) if h is not None else self.steps[i](element)
            logits.append(y)

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


model = RNN(CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)


training_iterations = 3000
for i in range(training_iterations):
    optimizer.zero_grad()
    batch = get_batch("train")
    outputs, loss = model(batch[0], batch[1])
    loss.backward()
    #print("Loss: ", loss.item())
    optimizer.step()


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


### Truly terrible results

output_file = open("output.txt", "w")

with torch.no_grad():
    hidden = None
    input_data = torch.randint(VOCAB_SIZE, (1, CONTEXT_SIZE)).float()
    for _ in range(1000):
        output, hidden = model(input_data, hidden)
        predicted_index = torch.argmax(output, dim=-1)
        predicted_char = int_to_string[predicted_index.item()]
        output_file.write(predicted_char)
        input_data = torch.cat((input_data[:, 1:], predicted_index.float()), dim=1)

output_file.close()