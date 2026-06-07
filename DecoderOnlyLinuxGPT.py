#This is all code written by hand following Andrej Karparthy's lecture " Let's build GPT: from scratch, in code, spelled out. "
#youtube link: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2799s
import torch.nn as nn
from torch.nn import functional as F
import torch
from typing import Any

"""
Notes:

You will not be able to run this on a macbook or something like that. Even on a spark it takes a long time (ie > 15 min).

This is a decoder only transformer.

we use the lower triangular mask to decode.

This is ~10 million parameter model trained on the linux code text file.


"""
batch_size = 64 #how many independent sequences to process in parallel
block_size = 256 #maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

#read in the file
with open('linux_input.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()


#all of the characters that occur in the set sorted
chars = sorted(list(set(text)))
vocab_size = len(chars)


#create mappings (simple encoder decoder maybe change to tiktoken)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder
decode = lambda l: ''.join([itos[i] for i in l]) #decoder


#encode the text and then store it in a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)


#Train, Validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#block size (context length)
x = train_data[:block_size]
y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     #print(f"when input is {context} the target: {target}")


#batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        #affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #aggregation
        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



#bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape


        tok_emb = self.token_embedding_table(idx) #(Batch,Time,Channels)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            #how well are we predicting the next character
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)#could also be -1
            loss = F.cross_entropy(logits, targets) #4.605170186 predicted, not the actual. Calculated from -ln(1/context)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range (max_new_tokens):
            idx_cond = idx[:, -block_size:]
            #getting predictions
            logits, loss = self(idx_cond)
            #focus on last step
            logits = logits[:, -1, :] #(B, C)
            #apply softmax for probs
            probs = F.softmax(logits, dim = 1) #(B, C)
            #sample
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #append to index in the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)
#print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#pytorch optimization 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb,yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
output = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(output)
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(output)
