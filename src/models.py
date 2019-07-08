import torch
from torch import nn
from torch.functional import F


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
    self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
    self.batchNorm1 = nn.BatchNorm2d(ndf * 2)
    self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
    self.batchNorm2 = nn.BatchNorm2d(ndf * 2)
    self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
    self.batchNorm3 = nn.BatchNorm2d(ndf * 8)
    self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False)
    self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    x = self.leakyReLU(self.conv1(x))
    x = self.leakyReLU(self.batchNorm1(self.conv2(x)))
    x = self.leakyReLU(self.batchNorm2(self.conv3(x)))
    x = self.leakyReLU(self.batchNorm3(self.conv4(x)))
    x = torch.sigmoid(self.conv5(x))
    return x


class Generator(nn.Module):
  def __init__(self, image_size, latent_size, hidden_size):
    super(Generator, self).__init__()
    self.convTranspose1(nz, ngf * 8, 4, 1, 0, bias=False)
    self.batchNorm1 = nn.BatchNorm2d(ndf * 8)
    self.convTranspose2(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
    self.batchNorm2 = nn.BatchNorm2d(ndf * 4)
    self.convTranspose3(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
    self.batchNorm3 = nn.BatchNorm2d(ndf * 2)
    self.convTranspose4(ngf * 2, ngf, 4, 2, 1, bias=False)
    self.batchNorm4 = nn.BatchNorm2d(ndf)
    self.convTranspose5(ngf, nc, 4, 2, 1, bias=False)
    self.ReLU = nn.ReLU(True)

  def forward(self, x):
    x = self.relu(self.batchNorm1(self.convTranspose1(x)))
    x = self.relu(self.batchNorm2(self.convTranspose2(x)))
    x = self.relu(self.batchNorm3(self.convTranspose3(x)))
    x = self.relu(self.batchNorm4(self.convTranspose4(x)))
    x = torch.tanh(self.convTranspose5(x))
    return x
