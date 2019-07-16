import torch
from torch import nn


class Discriminator(nn.Module):
  def __init__(self, nc, ndf):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
    self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(ndf * 2)
    self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(ndf * 4)
    self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(ndf * 8)
    self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
    self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    x = self.leakyReLU(self.conv1(x))
    x = self.leakyReLU(self.bn1(self.conv2(x)))
    x = self.leakyReLU(self.bn2(self.conv3(x)))
    x = self.leakyReLU(self.bn3(self.conv4(x)))
    x = torch.sigmoid(self.conv5(x))
    return torch.squeeze(x)


class Generator(nn.Module):
  def __init__(self, nz, ngf, nc):
    super(Generator, self).__init__()
    self.dconv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(ngf * 8)
    self.dconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(ngf * 4)
    self.dconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(ngf * 2)
    self.dconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
    self.bn4 = nn.BatchNorm2d(ngf)
    self.dconv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
    self.ReLU = nn.ReLU(True)  # inplace=True

  def forward(self, x):
    x = self.ReLU(self.bn1(self.dconv1(x)))
    x = self.ReLU(self.bn2(self.dconv2(x)))
    x = self.ReLU(self.bn3(self.dconv3(x)))
    x = self.ReLU(self.bn4(self.dconv4(x)))
    x = torch.tanh(self.dconv5(x))
    return x
