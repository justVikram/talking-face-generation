import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import encoder
import ContrastiveLoss
from dataset import random_dataset


encod = encoder.encoder()
learning_rate = 0.0001
optimizer = Adam(encod.parameters(), lr=learning_rate)
criterion = ContrastiveLoss.CL()

train_loader = random_dataset()
sum = 0
for param in encod.parameters():
    sum += param.numel()
    print(param.numel())
print("____________________________")
print(sum)
epochs = 1


losses = []
for epoch in range(epochs):
    for i, (image, audio) in enumerate(train_loader):
        i += 1
        im_embed, au_embed = encod(image, audio)

        loss = criterion(im_embed, au_embed)

        if i % 2 == 0:
            print(f"Epoch{epoch+1:6}    loss{loss:10.4f}%    batch{i:6}")
            
           

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss)
