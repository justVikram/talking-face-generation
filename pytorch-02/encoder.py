
import torch
import torch.nn as nn
import torch.nn.functional as F



test_image = torch.rand(size=(1, 3, 128, 256))

test_audio = torch.rand(size=(1, 1, 257, 199))


class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.im_conv_1 = nn.Conv2d(3, 16, 5, 1)
        self.im_conv_2 = nn.Conv2d(16, 32, 4, 1)
        self.im_conv_3 = nn.Conv2d(32, 64, 4, 1)
        self.im_conv_4 = nn.Conv2d(64, 128, 3, 1)
        self.im_conv_5 = nn.Conv2d(128, 256, 3, 1)
        self.im_conv_6 = nn.Conv2d(256, 512, 2, 1)
        self.im_fc1 = nn.Linear(2560, 1024)
        self.im_fc2 = nn.Linear(1024, 256)

        self.au_conv_1 = nn.Conv2d(1, 64, 3)
        self.au_conv_2 = nn.Conv2d(64, 128, 3)
        self.au_conv_3 = nn.Conv2d(128, 256, 3)
        self.au_conv_4 = nn.Conv2d(256, 512, 3)
        self.au_fc1 = nn.Linear(512, 256)

    def forward(self, x, y):
        x = F.leaky_relu(self.im_conv_1(x))
        x = F.max_pool2d(x, 2)

        y = F.leaky_relu(self.au_conv_1(y))
        y = F.max_pool2d(y, 2)

        x = F.leaky_relu(self.im_conv_2(x))
        x = F.max_pool2d(x, 2)

        y = F.leaky_relu(self.au_conv_2(y))
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_3(x))
        x = F.max_pool2d(x, 2)

        y = F.leaky_relu(self.au_conv_3(y))
        y = F.max_pool2d(y, 2)

        x = F.leaky_relu(self.im_conv_4(x))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.im_conv_5(x))

        y = F.leaky_relu(self.au_conv_4(y))
        y = F.max_pool2d(y, (28, 21))

        x = F.leaky_relu(self.im_conv_6(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 2560)

        y = y.view(-1, 512)
        y = self.au_fc1(y)
        
        x = F.leaky_relu(self.im_fc1(x))
        x = self.im_fc2(x)

        y = y.view(-1)
        x = x.view(-1)
        return x, y


