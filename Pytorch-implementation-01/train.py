import torch
from torch.optim import Adam
from dataset import random_dataset
import encoder
from pytorch_metric_learning import losses


encod          = encoder.encoder()
learning_rate  = 0.0001
optimizer      = Adam(encod.parameters(),lr=learning_rate)
criterion      = losses.ContrastiveLoss()

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
        print(batch_counter)
        #loss = criterion(im_embed,au_embed)

        #fix the error here

        optimizer.zero_grad()
        #loss.backward()
        optimizer.step()
    #losses.append(loss)
        
        
    


    
    
