# pytorch-DCGAN
## Description
DCGANのpytorch実装

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
