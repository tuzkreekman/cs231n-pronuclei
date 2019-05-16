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
    parser.add_argument("-num_epochs", help="number of output classes", default=25)
    parser.add_argument("-freeze", help="freeze weights while training", default="False")
    parser.add_argument("-bsize", help="batch size", default=4)
    args = parser.parse_args()

    return args.data, float(args.lr), int(args.out_classes), int(args.num_epochs), bool(args.freeze), int(args.bsize)

def transform_data(data_path, b_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-45, 45)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    model_ft = models.vgg16_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    features = list(model_ft.classifier.children())[:-1]
    if frozenWeights:
        for param in model_ft.features.parameters():
            param.require_grad = False
    features.extend([nn.Linear(num_ftrs, outFeatures)])
    model_ft.classifier = nn.Sequential(*features)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learn_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler
def main():
    dataPath, learn_rate, out_classes, num_epochs, freeze_weights, batch_size = parse_args()
    dataloaders, dataset_sizes, class_names, device = transform_data(dataPath, batch_size)
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = makeModel(device, learn_rate,
                                                                    True, out_classes, freeze_weights)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs, dataloaders, device, dataset_sizes, class_names)
if __name__ == "__main__":
    main()
