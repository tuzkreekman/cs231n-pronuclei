# This code was adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# it was modified for plotting masks with the image, 
# and to support the variable mask sizes we have 
# for different segmentation algorithms

import numpy as np
import matplotlib.pyplot as plt
import torch

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp /= 255
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def imshowimagemasked(img, mask, title=None, plot_original=False):
    """Imshow for Tensor."""
    inp = img
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if plot_original:
        plt.imshow(inp)
    plt.imshow(mask.numpy(), alpha=.2) #,cmap='jet',alpha=0.2)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def visualize_segmenter(model, dataloader, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            is_deeplab = labels.shape[-1]==65
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                imshow(inputs.cpu().data[j])
                
                imshowimagemasked(inputs.cpu().data[j], preds.cpu().data[j], plot_original=is_deeplab)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)