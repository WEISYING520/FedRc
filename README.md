# FedRC: Coordinating Knowledge Sharing in Federated Learning Assisted Mobile Computing

Implementation of the paper accepted by TMC 2026 : .

Supplement to the FedRC training process
## Requirments
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5

## Data Preparation
We can automatically download and split the dataset using the following code.

* Generate training data on MNIST
```
python generate_data.py --dataset mnist --n_class 10 --alpha 0.5
```
* Generate training data on CIFAR10
```
python generate_data.py --dataset cifar10 --n_class 10 --alpha 0.5
```
* Generate training data on CIFAR100
```
python generate_data.py --dataset cifar100 --n_class 100 --alpha 0.5
```

## Running the experiments
We provide two running options.

Option 1: You can directly input running parameters via the command line. In this scenario, you need to comment out the call to the `prepare_args` function in the `main_*.py` file. The following is an example:
* To train the FedRC on MNIST:
```
python main_h.py --R 120  --lr 0.005 --kd y --dataset mnistu10c10-alpha0.05-ratio0.5 --save_name rc_h_alpha0.05
```
* To train the FedRC on CIFAR10:
```
python main_h.py --R 120  --lr 0.005 --kd y --dataset cifar10u10c10-alpha1.0-ratio0.5 --save_name rc_h_cifar10_alpha1.0
```
* To train the FedRC on CIFAR100:
```
python main_h.py --R 120  --lr 0.005 --kd y --dataset cifar100u10c10-alpha1.0-ratio0.5 --classes 100 --save_name rc_h_cifar100_alpha1.0
```
Option 2: You can directly set the running parameters by modifying the parameters in the `def prepare_args(args):` function in `main_*.py`.

```
def prepare_args(args):
    args.R = 10
    args.lr = 0.005
    args.classes = 10
    # args.dataset = 'mnistu10c10-alpha0.05-ratio0.5'
    args.device = 'cuda:0'
    args.global_device = 'cuda:0'

    args.kd = 'y'
    args.save_name = 'FedRC_time_10'
```
Run `main_*.py` after modifying `def prepare_args(args):`.
```
python main_*.py
```
## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: mnistu10c10-alpha0.05-ratio0.5
* ```--device:```   Default: 'cuda:0'. Options: 'cuda', 'cpu'
* ```--R:```        Number of rounds. Default is 120.
* ```--classes:```  Number of classes. Default is 10.

#### Total Parameters
* ```--node_num:```  Number of nodes. Default is 10.
* ```--E:```         Number of local epochs. Default is 5.
* ```--notes:```     Notes of Experiments. Default is empty string.
* 
#### Optima Parameters
* ```--optimizer:``` Optimizer. Default: 'sgd'. Options: 'sgd', 'adam'
* ```--lr:```        Learning rate set to 0.005 by default.
* ```--lr_step:```   Learning rate decay step size. Default is 10.
* ```--stop_decay:``` Round when learning rate stop decay. Default is 50.
* ```--momentum:```   SGD momentum. Default is 0.9.
* ```--alpha:```      Local ratio of data loss. Default is 0.5.
* ```--beta:```       Meme ratio of data loss. Default is 0.5.

#### Debug Parameters
* ```--overlap_ratio:```    Overlap ratio. Default is 0.8.
* ```--select_node:```      Select node count. Default is 10.
* ```--save_name:```        Save name. Default is 'main'.
* ```--kd:```               Knowledge distillation. Default is 'y'.
* ```--time_slot_rounds:``` Time slot rounds. Default is 50.
* ```--time_slot:```        Time slot. Default is 8.



## Citation
If you find this project helpful, please consider to cite the following paper:
```
补充blib
```
