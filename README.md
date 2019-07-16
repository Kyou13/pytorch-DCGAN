# pytorch-DCGAN
## Description
DCGANのpytorch実装

### DCGAN
[papaer link](https://arxiv.org/abs/1511.06434)
- GANに畳み込み層を導入

## Example
### loss
![loss](https://github.com/Kyou13/pytorch-DCGAN/blob/master/samples/mnist/loss.png)
### Genarated Image
- epochs: 5
  - batch size: 128

![genaratedImage](https://github.com/Kyou13/pytorch-DCGAN/blob/master/samples/mnist/fake_images_190717032103.png)


## Requirement
- Python 3.7
- pytorch 1.1.0
- torchvision 0.3.0
- Click

## Usage
### Training
```
$ pip install -r requirements.txt 
$ python main.py train [--dataset]
# training log saved at ./samples/fake_images-[epoch].png
```

### Generate
```
$ python main.py generate [--dataset]
# saved at ./samples/fake_images_%y%m%d%H%M%S.png
```
