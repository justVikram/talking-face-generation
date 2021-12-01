from dataready import crop 
import encoder
import cv2 as cv
import torch

encod = encoder.encoder()

img = cv.imread("input.png")
img = crop(img)

y = torch.rand(size=(1, 1, 257, 199))
x,y = encod(img, y)
print(x.view(-1).shape,y.view(-1).shape)
print(x,y)