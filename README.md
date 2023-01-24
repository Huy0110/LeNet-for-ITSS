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

## On remote
from your local machine, run
```
ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
```
on the remote machine, run:
```
tensorboard --logdir <path> --port 6006
```
Then, navigate to (in this example) http://localhost:16006 on your local machine.

### Note
If you don't have data yet, please add 'download=True' into MNIST function in dataloader_mnist.py to automately download the dataset