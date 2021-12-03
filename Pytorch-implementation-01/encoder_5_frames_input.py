
import torch
import torch.nn as nn
import torch.nn.functional as F


# 5 frames per input 1->pointer in dataloader 3->channels 5->no of images 128->height 255->width
test_image = torch.rand(size=(1, 3, 5, 128, 255))
# same input dimensions like in L3 architecture
test_audio = torch.rand(size=(1, 1, 257, 199))
class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):

        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))


        return loss

class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.im_conv_1 = nn.Conv3d(3,5,(5,7,7),1,(1,1,1))
        self.im_conv_2 = nn.Conv3d(5,7,(3,3,3),1,(1,1,1))
        self.im_conv_3 = nn.Conv3d(7,9,(2,2,2),1,(1,1,1))
        self.im_conv_4 = nn.Conv3d(9,11,(1,1,1),1,(1,1,1))
        self.im_conv_5 = nn.Conv3d(11,13,(2,2,1),1,0)
        self.im_conv_6 = nn.Conv3d(13,15,(2,2,1),1,0)
        self.fc1 = nn.Linear(360, 256)

        self.au_conv_1 = nn.Conv2d(1, 64, 3)
        self.au_conv_2 = nn.Conv2d(64, 128, 3)
        self.au_conv_3 = nn.Conv2d(128, 256, 3)
        self.au_conv_4 = nn.Conv2d(256, 512, 3)
        self.au_fc1 = nn.Linear(512, 256)

    def forward(self, x, y):
        x = F.leaky_relu(self.im_conv_1(x))
        y = F.leaky_relu(self.au_conv_1(y))
        x = F.max_pool3d(x, (1, 3, 3))
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_2(x))
        y = F.leaky_relu(self.au_conv_2(y))
        x = F.max_pool3d(x, (1, 3, 3))
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_3(x))
        y = F.leaky_relu(self.au_conv_3(y))
        
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_4(x))
        x = F.leaky_relu(self.im_conv_5(x))
        x = F.max_pool3d(x,(1,3,3))
        x =  F.leaky_relu(self.im_conv_6(x))
        x =  F.max_pool3d(x,(1,2,3))
        x = x.view(-1, 360)
        x = F.leaky_relu(self.fc1(x))
        y = F.leaky_relu(self.au_conv_4(y))
        y = F.max_pool2d(y, (28, 21))
        y = y.view(-1, 512)
        y = self.au_fc1(y)
        x = x.view(-1)
        y = y.view(-1)
        return x, y


model = encoder()


print(test_image.shape, test_audio.shape)


img_embed, audio_embed = model(test_image, test_audio)


print(img_embed.detach().numpy().shape, audio_embed.detach().numpy().shape)

cl = ContrastiveLoss()

print(cl(img_embed,audio_embed))