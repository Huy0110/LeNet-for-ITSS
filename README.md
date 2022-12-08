# LeNet

## Setup

Install all dependencies using the following command

```
$ pip install -r requirements.txt
```

## Usage

### Train

```
$ python train.py
```

### Test

```
$ python test.py
```

### Visualize
Run this command and you will get your trainning process visualized at http://localhost:6006
```
$ tensorboard --logdir=runs
```

### Note
If you don't have data yet, please add 'download=True' into MNIST function in dataloader_mnist.py to automately download the dataset