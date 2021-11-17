"""
DESCRIPTION:    this file contains all the functions used for training network with early-exit layers

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

print("I'm very vegetable. I know nothing about what I'm doing right now, and currently I \
only want to touch fish and hua water.")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from nikolaos import cifar
import utils
import model as my_models


# optimizer = optim.SGD(model.parameters(), lr=0.0005)


def train( model, optimizer, train_loader, test_loader, criterion, epoch, early_exit, mode, args ):
    '''
    if mode = 'all', then it will train the parameters without any early-exit mechanisms
    if mode = 'exit1', then it will train the parameters for the fitst early-exit layer
    if mode = 'exit2', then it will train the parameters for the second early-exit layer
    if mode = 'exit3', then it will train the parameters for the third early-exit layer
    '''
    if mode == 'all':
        model.set_exit_layer( -1 )
    elif mode == 'exit1':
        model.set_exit_layer( 0 )
    elif mode == 'exit2':
        model.set_exit_layer( 1 )
    elif mode == 'exit3':
        model.set_exit_layer( 2 )
    elif mode == 'all_and_exit':
        model.set_exit_layer( -2 )
    else:
        raise NotImplementedError
    best_test_acc = 0
    num_no_increase = 0
    model.print_average_activations()
    for epoch_idx in range(epoch):
        if num_no_increase >= 5 and early_exit:
            print( 'early exit is triggered' )
            return
        print(f'\nEpoch: {(epoch_idx + 1)}')
        model.train()
        train_loss = 0
        correct = 0
        correct_exit1, correct_exit2, correct_exit3, correct_all = 0, 0, 0, 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            if mode != 'all_and_exit':
                outputs = model( images )
                loss = criterion( outputs, labels )
            else:
                exit1, exit2, exit3, all = model( images )
                loss =  criterion( exit1, labels ) + \
                        criterion( exit2, labels ) + \
                        criterion( exit3, labels )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if mode != 'all_and_exit':
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f |  Acc: %.3f%% (%d/%d)' %
                        (epoch_idx + 1, batch_idx + 1, train_loss / 2000, 100.*correct/total, correct, total))
                    train_loss = 0.0
                    total = 0
                    correct = 0
            else:
                _, exit1_predicted = exit1.max(1)
                _, exit2_predicted = exit2.max(1)
                _, exit3_predicted = exit3.max(1)
                _, all_predicted = all.max(1)
                total += labels.size(0)
                correct_exit1 += exit1_predicted.eq(labels).sum().item()
                correct_exit2 += exit2_predicted.eq(labels).sum().item()
                correct_exit3 += exit3_predicted.eq(labels).sum().item()
                correct_all += all_predicted.eq(labels).sum().item()
                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f ||  Acc: exit1: %.3f%% (%d/%d) | exit2: %.3f%% (%d/%d) | exit3: %.3f%% (%d/%d) | all: %.3f%% (%d/%d)' %
                        (epoch_idx + 1, batch_idx + 1, train_loss / 2000, 
                        100.*correct_exit1/total, correct_exit1, total,
                        100.*correct_exit2/total, correct_exit2, total,
                        100.*correct_exit3/total, correct_exit3, total,
                        100.*correct_all/total, correct_all, total))
                    train_loss = 0.0
                    total = 0
                    correct_exit1, correct_exit2, correct_exit3, correct_all = 0, 0, 0, 0
        if epoch_idx % 5 == 4:
            print('begin middle test:')
            current_test_acc = test( model, mode, test_loader, False, args )
            model.print_average_activations()
            if mode != 'all_and_exit':
                if current_test_acc > best_test_acc:
                    best_test_acc = current_test_acc
                    num_no_increase = 0
                else:
                    num_no_increase += 1



def test( model, mode, test_loader, verbose, args ):
    if mode == 'all':
        model.set_exit_layer( -1 )
    elif mode == 'exit1':
        model.set_exit_layer( 0 )
    elif mode == 'exit2':
        model.set_exit_layer( 1 )
    elif mode == 'exit3':
        model.set_exit_layer( 2 )
    elif mode == 'all_and_exit':
        model.set_exit_layer( -2 )
    else:
        raise NotImplementedError
    model.eval()
    if mode != 'all_and_exit':
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        general_acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    else:
        correct_exit1, correct_exit2, correct_exit3, correct_all = 0, 0, 0, 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(args.device), labels.to(args.device)
                exit1, exit2, exit3, all = model( images )
                _, exit1_predicted = exit1.max(1)
                _, exit2_predicted = exit2.max(1)
                _, exit3_predicted = exit3.max(1)
                _, all_predicted = all.max(1)
                total += labels.size(0)
                correct_exit1 += exit1_predicted.eq(labels).sum().item()
                correct_exit2 += exit2_predicted.eq(labels).sum().item()
                correct_exit3 += exit3_predicted.eq(labels).sum().item()
                correct_all += all_predicted.eq(labels).sum().item()
        print('Accuracy of the network on the 10000 test images: \nexit1: %d %% | exit2: %d %% | exit3: %d %% | all: %d %%' %
            (100.*correct_exit1/total,
            100.*correct_exit2/total,
            100.*correct_exit3/total,
            100.*correct_all/total, ))

    if mode != 'all_and_exit' and verbose:
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
    if mode != 'all_and_exit':
        return general_acc

def train_all(model, train_loader, test_loader, criterion, args):
    optimizer_name = 'adam'
    epoch_num = 100
    learning_rate = 0.0005
    mode = 'all'
    print(f'\n\noptimizer: {optimizer_name}\nbatch size: {args.batch_size}\nepoch number: {epoch_num}\nlearning rate: {learning_rate}\ntraining mode: {mode}\n\n')

    print('the parameter names and shapes for model is:')
    for name, parameter in model.named_parameters():
        print(f'name: {name}, parameter shape: {parameter.shape}')

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    train( model, optimizer, train_loader, test_loader, criterion, epoch_num, mode=mode, early_exit=True, args=args )
    test(model, mode, test_loader, True, args)
    if args.download_model:
        torch.save( model, './models/trained_all.pt' )

def train_exit_simultaneously(model, train_loader, test_loader, criterion, args):
    optimizer_name = 'adam'
    epoch_num = 100
    learning_rate = 0.001
    mode = 'all_and_exit'
    print(f'\n\noptimizer: {optimizer_name}\nbatch size: {args.batch_size}\nepoch number: {epoch_num}\nlearning rate: {learning_rate}\ntraining mode: {mode}\n\n')

    model = torch.load( './models/trained_all.pt' )
    model = model.to(args.device)

    print('the parameter names and shapes for model is:')
    for name, parameter in model.named_parameters():
        print(f'name: {name}, parameter shape: {parameter.shape}')

    for name, parameter in model.named_parameters():
        if name not in args.early_exit_names:
            parameter.requires_grad = False

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError
    
    print('now testing the accuracy for mode all_and_exit')
    test( model, mode, test_loader, True, args )
    
    train( model, optimizer, train_loader, test_loader, criterion, epoch=epoch_num, mode=mode, early_exit=True, args=args )
    # model, optimizer, train_loader, test_loader, epoch, early_exit, mode
    test( model, mode, test_loader, True, args )
    if args.download_model:
        torch.save( model, './models/trained_all_simultaneously.pt' )

    print('now testing the accuracy for mode all_and_exit')
    test(model, mode, test_loader, True, args)


def train_and_save_early_exit_network( args:utils.Namespace ):
    model = cifar.CIFAR_NET_Light()
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=4)
    print(args.device)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()

    if args.method == 'all':
        train_all(model, train_loader, test_loader, criterion, args)
    elif args.method == 'early_exit':
        train_exit_simultaneously(model, train_loader, test_loader, criterion, args)
    elif args.method == 'all_and_early_exit':
        train_all(model, train_loader, test_loader, criterion, args)
        train_exit_simultaneously(model, train_loader, test_loader, criterion, args)

def test_normal( model, test_loader, verbose, device ):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    general_acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    if verbose:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(labels.shape.numel()):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
    return general_acc

def train_normal():
    # training setup
    optimizer_name = 'adam'
    epoch_num = 100
    learning_rate = 0.0005
    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = my_models.CIFAR_Normal().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=4)
    # begin training
    best_test_acc = 0
    num_no_increase = 0
    for epoch_idx in range(epoch_num):
        if num_no_increase >= 5:
            print( 'early exit is triggered' )
            return
        print(f'\nEpoch: {(epoch_idx + 1)}')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model( images )
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.5f |  Acc: %.3f%% (%d/%d)' %
                    (epoch_idx + 1, batch_idx + 1, train_loss / 2000, 100.*correct/total, correct, total))
                train_loss, total, correct = 0.0, 0, 0
        if epoch_idx % 5 == 4:
            print('begin middle test:')
            current_test_acc = test_normal( model, test_loader, False, device )
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                num_no_increase = 0
            else:
                num_no_increase += 1
    test_normal( model, test_loader, True, device )
    torch.save( model, './models/trained_normal.pt' )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model', default='' )
    pass

if __name__ == '__main__':
    # TODO: add namespace args
    args = utils.Namespace()
    args.batch_size = 128
    args.method = 'all_and_early_exit'
    args.early_exit_names = ['exit_1.a', 'exit_1.c', 'exit_1.n1', 'exit_1.n2', 'exit_1.codebook.weight', 
                             'exit_1.codebook.bias', 'exit_1_fc.weight', 'exit_1_fc.bias', 'exit_2.a', 
                             'exit_2.c', 'exit_2.n1', 'exit_2.n2', 'exit_2.codebook.weight', 'exit_2.codebook.bias']
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
    args.download_model = True
    train_and_save_early_exit_network( args )
    # train_normal()