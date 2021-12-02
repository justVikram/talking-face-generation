import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import encoder_5_frames_input
import ContrastiveLoss
from SecondEncoderDataset import random_dataset



encod          = encoder_5_frames_input.encoder()
learning_rate  = 0.0001
optimizer      = Adam(encod.parameters(),lr=learning_rate)
criterion      = ContrastiveLoss.CL()

train_loader = random_dataset()
sum = 0
for param in encod.parameters():
    sum += param.numel()
    print(param.numel())
print("____________________________")
print(sum)
epochs = 1
batch_counter = 0

losses = []
for epoch in range(epochs):
    for image,audio in train_loader:
        im_embed,au_embed = encod(image,audio)
        batch_counter += 1
        
        loss = criterion(im_embed,au_embed)

        print(f"Epoch{epoch+1} loss{loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    losses.append(loss)