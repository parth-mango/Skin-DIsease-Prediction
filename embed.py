import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class ResNet152Embedder(nn.Module):
    """Grabs the average pool embedding layer from ResNet152, 
    which seems to do very nicely in organizing images.
    """
    def __init__(self):
        super(ResNet152Embedder, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        self.embedder = nn.Sequential(*(list(resnet152.children())[:-1]))

    def forward(self, x):
        return self.embedder(x)
