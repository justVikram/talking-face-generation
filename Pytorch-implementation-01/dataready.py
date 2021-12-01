import torch
def crop(img):
    lower_half = img[128:,:,:]
    return torch.Tensor(lower_half).view(1,3,128,256)
