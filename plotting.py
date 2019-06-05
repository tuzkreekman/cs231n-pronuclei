# This code was adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# it was modified for plotting masks with the image, 
# and to support the variable mask sizes we have 
# for different segmentation algorithms

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.misc import imresize
import cv2

def imshow(inp):
    """Imshow for Tensor."""
    inp /= 255
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def imshowimagemasked(img, mask, title=None):
    """Imshow for Tensor with overlayed masks."""
    inp = img/255
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.imshow(imresize(mask.numpy(), inp.shape), cmap='jet',alpha=0.2)
    plt.axis('off')
    if title!=None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def visualize_segmenter(model, dataloader, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)[:,0]

            for j in range(inputs.size()[0]):
                images_so_far += 1
                imshow(inputs.cpu().data[j])
                imshowimagemasked(inputs.cpu().data[j], preds.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def pretty_plot_segmenter(model, dataloader, device, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)[:,0]

            for j in range(inputs.size()[0]):
                images_so_far += 1
                imshowimagemasked(inputs.cpu().data[j], preds.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
def visualize_counting_errors(model, dataloader, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    
    f, axs = plt.subplots(num_images,3,figsize=(5,5.0*num_images/3),sharex='col',sharey='col')

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = np.squeeze(torch.sigmoid(outputs).data.cpu().numpy()[:,0])
            grounds = np.squeeze(labels.data.cpu().numpy()[:,0])

            for j in range(inputs.size()[0]):
                bw_p = np.array(255*(preds[j]>.5), np.uint8)
                bw_g = np.array(255*(grounds[j]>.5), np.uint8)
                nlabels, _, _, _ = cv2.connectedComponentsWithStats(bw_p)
                counts_p = nlabels - 1
                nlabels, _, _, _ = cv2.connectedComponentsWithStats(bw_g)
                counts_g = nlabels - 1

                if counts_p!=counts_g:
                    images_so_far += 1


                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))/255
                    ax = axs[images_so_far-1, 0]
                    ax.imshow(inp)
                    ax.axis('off')
                    if images_so_far==1:
                        ax.set_title('Original')
    
                    ax = axs[images_so_far-1, 1]
                    ax.imshow(bw_p)
                    ax.axis('off')
                    if images_so_far==1:
                        ax.set_title('Predictions')
                    
                    
                    ax = axs[images_so_far-1, 2]
                    ax.imshow(bw_g)
                    ax.axis('off')
                    if images_so_far==1:
                        ax.set_title('Ground Truth')
                    
                    #print('Really have',counts_g,'but predicted',counts_p)

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        plt.pause(0.001)

                        return
        plt.pause(0.001)

        model.train(mode=was_training)