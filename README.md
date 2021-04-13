# pointer-generator-model-pytorch

This is a pointer-generator network model built with PyTorch. Yes, you noticed that the entire format is extremely close to this [Tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb). We modified only the necessary parts to add the pointer mechanism and coverage according to the original [paper](https://www.aclweb.org/anthology/P17-1099/).

To use this code, we sincerely recommend you to take a look at this [Tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb). You can just import the model and replace the one in the tutorial.

As everyone may have their own application with this model, we would not publish a separate file for the usage, but rather paste it here. This usage is also very similar to the format you saw in the tutorial just now. One **BIG** difference you need to notice is that the original ```Field``` class instances are not enough to handle out-of-vocabulary words, and thus all that you need to do is to put the ```oov.py``` file from [here](https://cestwc.medium.com/how-do-you-write-a-clean-pointer-generator-model-with-pytorch-80d25bde113b) into the directory where your codes are running. You can assume this ```oov.py``` as a patch to standard Torchtext 0.9.1, till the day newer versions emerge. Another **difference** is that the loss function has to be the negative log likelihood loss, as the last layer from ```Decoder``` is already in a form of probablity (which you need to ```torch.log``` is later before calculating the loss).

## Usage

```python
from model import Attention, Encoder, Decoder, Seq2Seq

INPUT_DIM = len(YOUR_SRC_FIELD.vocab)
OUTPUT_DIM = len(YOUR_TRG_FIELD.vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = torch.nn.NLLLoss(ignore_index = TRG_PAD_IDX)
```

### model training and evaluating
```python
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src, src_len = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, repetition = model(src, src_len, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(torch.log(output), trg)
        
        alpha = 0.2 if loss < 1.66 else 1 if loss < 1.8 else 0
		
        loss_with_coverage = loss + repetition * alpha
        
        loss_with_coverage.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output, _ = model(src, src_len, trg, 0) #turn off teacher forcing
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(torch.log(output), trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 3
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut4-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
```

### Inference 
you may need two more functions than the original tutorial
```python
def stoiOOV(arr, srcArr, vocab):
    ids = []
    srcOOV = list(set([x for x in srcArr if vocab.stoi[x] == vocab.stoi['<unk>']]))
    for i in range(len(arr)):
        idx = vocab.stoi[arr[i]]
        if idx == vocab.stoi['<unk>'] and arr[i] in srcOOV:
            idx = len(vocab) + srcOOV.index(arr[i]) # Map to its temporary article OOV number
        ids.append(idx)
    return ids

def itosOOV(ids, srcArr, vocab):
    arr = []
    srcOOV = list(set([x for x in srcArr if vocab.stoi[x] == vocab.stoi['<unk>']]))
    for i in range(len(ids)):
        if ids[i] < len(vocab):
            string = vocab.itos[ids[i]]
        elif (ids[i] - len(vocab)) < len(srcOOV):
            string = srcOOV[ids[i] - len(vocab)]
        else:
            string = '<unk>'
        arr.append(string)
    return arr
```
and then
```python
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = stoiOOV(tokens, tokens, src_field.vocab)
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    coverage = Variable(torch.zeros(mask.shape)).to(mask.device)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, coverage, attention, _ = model.decoder(trg_tensor, hidden, encoder_outputs, mask, coverage, src_tensor)

        attentions[i] = attention
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = itosOOV(trg_indexes, tokens, trg_field.vocab)
    
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]
```

## Final words
Again, the majority of the **Usage** codes are from this [Tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb). Many useful functions like ```display_attention``` are not re-displayed here as we did not change at all. Stacking the **Usage** codes from the [OOV](https://cestwc.medium.com/how-do-you-write-a-clean-pointer-generator-model-with-pytorch-80d25bde113b) repository and the ones here should be enough to get your task done.

Don't forget to import necessary packages like these

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
```
