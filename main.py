"""
DESCRIPTION:    this file contains the main program

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import torch
import torchvision
import torchvision.transforms as transforms
import time

from nikolaos import cifar
from train import test
import utils

args = utils.Namespace()
args.batch_size = 128
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

model = cifar.CIFAR_NET_Light().to(args.device)
model = torch.load( './models/trained_all_simultaneously.pt' ).to(args.device)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, num_workers=4)

# first test original time
mode = 'all'
st_time = time.perf_counter()
test( model, mode, test_loader, verbose=True, args=args )
end_time = time.perf_counter()
print( f'time consumed (all): {end_time - st_time}' )

# then test exit 1 time
mode = 'exit1'
st_time = time.perf_counter()
test( model, mode, test_loader, verbose=True, args=args )
end_time = time.perf_counter()
print( f'time consumed (exit1): {end_time - st_time}' )

# next test exit 2 time
mode = 'exit2'
st_time = time.perf_counter()
test( model, mode, test_loader, verbose=True, args=args )
end_time = time.perf_counter()
print( f'time consumed (exit2): {end_time - st_time}' )

# finally test exit 3 time
mode = 'exit3'
st_time = time.perf_counter()
test( model, mode, test_loader, verbose=True, args=args )
end_time = time.perf_counter()
print( f'time consumed (exit3): {end_time - st_time}' )