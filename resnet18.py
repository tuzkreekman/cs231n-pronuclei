# from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import *
plt.ion()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", help="path to data file")
    parser.add_argument("-lr", help="learning rate", default=1e-7)
    parser.add_argument("-out_classes", help="number of output classes", default=5)
    parser.add_argument("-num_epochs", help="number of epochs", default=25)
    parser.add_argument("-freeze", help="freeze weights while training (must be yes/no)", default='no')
    parser.add_argument("-bsize", help="batch size", default=128)
    parser.add_argument("-savePath", help="path to save pickle of best model", default='./models/lastResnet.pt')
    args = parser.parse_args()
    if args.freeze == 'yes':
        freeze = True
    elif args.freeze == 'no':
        freeze = False
    else:
        freeze = -1
    return args.data, float(args.lr), int(args.out_classes), int(args.num_epochs), freeze, int(args.bsize), args.savePath

def transform_data(data_path, b_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-45, 45)),
            transforms.Resize((224,224)),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    path2Data = data_path
    image_datasets = {x: datasets.ImageFolder(os.path.join(path2Data, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=b_size,
                                                  shuffle=True, num_workers=8)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes, class_names, device

def imshow(dataloaders, class_names, inp):
    """Imshow for Tensor."""
    inputs, classes = next(iter(dataloaders['train']))
    inp = torchvision.utils.make_grid(inputs)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.interactive(False)
    plt.imshow(inp)
    plt.show()
    plt.title([class_names[x] for x in classes])
    plt.pause(0.001)  # pause a bit so that plots are updated
    return

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, device, dataset_sizes, class_names):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_labels = []
            running_preds = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_labels.extend(list(labels.data.cpu()))
                running_preds.extend(list(preds.cpu()))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            cm = makeConfusionMatrix(np.array(running_labels, dtype=np.int32), np.array(running_preds, dtype=np.int32))
            print(cm)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            sens, spec = getSensitivityAndSpecificity(cm)
            print('Sensitivity: {:.4f} Specificity {:.4f}'.format(sens, spec))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_labels = running_labels.copy()
                best_preds = running_preds.copy()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Confusion matrix')
    plot_confusion_matrix(makeConfusionMatrix(np.array(best_labels, dtype=np.int32),np.array(best_preds, dtype=np.int32)), class_names)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def makeModel(device, learn_rate, pretrained=True, outFeatures=5, frozenWeights=False):
    model_conv = torchvision.models.resnet18(pretrained=True)
    #print("Freezing weights: %s" %frozenWeights) 
    if frozenWeights:
        for param in model_conv.parameters():
                param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, outFeatures)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learn_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    return model_conv, criterion, optimizer_conv, exp_lr_scheduler

def main():
    dataPath, learn_rate, out_classes, num_epochs, freeze_weights, batch_size, savePath = parse_args()
    #print(freeze_weights)
    dataloaders, dataset_sizes, class_names, device = transform_data(dataPath, batch_size)
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = makeModel(device, learn_rate,
                                                                    True, out_classes, freeze_weights)
    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs, dataloaders, device, dataset_sizes, class_names)
    torch.save(model, savePath)
if __name__ == "__main__":
    main()
