"""
DESCRIPTION:    this file contains all the functions to apply adaptive early-exit mechanisms
                during the inference process of the network

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import torch
import torchvision
import torchvision.transforms as transforms

from nikolaos import cifar
import model as my_models
from train import test

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, transform=transform, download=False)

model_early_exit_temp = cifar.CIFAR_NET_Light()
model_early_exit = my_models.CIFAR_Adaptive( [0, 0, 0], 1 )
model_normal = my_models.CIFAR_Normal()

model_early_exit_temp = torch.load( './models/trained_all_simultaneously.pt' )

model_normal = torch.load( './models/trained_normal.pt' )

# 1. calculat the average activations
# 2. transfer the parameter in model_early_exit_temp to model_early_exit

# 3. try change horizon_split to correct value
# 4. try turning off the training for exit2 alone
# 5. try divicing exit_3 by 2

model_early_exit.load_state_dict( model_early_exit_temp.state_dict() )

# test the accuracy
# test the runtime

model_early_exit.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model_early_exit(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

general_acc = 100 * correct / total
print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(args.device), labels.to(args.device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.shape.numel()):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        args.classes[i], 100 * class_correct[i] / class_total[i]))