from PIL import Image
import torch
import os

# In own repository
# other's code
from segmentation_utils import *
# adapted code
from plotting import *
# own code
from scoring import iou

def score_segmenter(model, dataloaders, device):
    was_training = model.training
    model.eval()
    running_score=0
    num_imgs = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            #print(iou(outputs, labels),labels.shape)
            running_score += iou(outputs, labels).sum() 
            num_imgs += inputs.shape[0]
        
        running_score /= num_imgs #len(dataloaders['val'])
        model.train(mode=was_training)
    return running_score

class Dataset(object):
    def __init__(self, datapath, image_size=224, mask_size=224, is_deeplab=False):
        # load paths for your dataset and put in self.data
        self.datapath = datapath
        self.maskpath = '../masks'
        self.image_size = image_size
        self.mask_size = mask_size
        self.is_deeplab = is_deeplab
        
        self.data = []
        self.masks = []
        for folder in os.listdir(datapath):
            for filename in os.listdir(datapath+'/'+folder):
                self.data.append(folder+'/'+filename)
                self.masks.append(filename.split('.')[0])
            
            
    def __getitem__(self, idx):
        im_dim = self.image_size
        
        # load images
        image = Image.open(self.datapath + '/' +  self.data[idx]).convert("RGB").resize((im_dim,im_dim))
        image_size = (image.size[0], image.size[1], 3)
        image = np.array(image.getdata()).reshape(image_size)
        maskname = self.maskpath + '/' +  self.masks[idx] + '_mask.jpg'
        mask_size = (self.mask_size,self.mask_size)

        if self.masks[idx] + '_mask.jpg' in os.listdir(self.maskpath):
            mask = np.array(Image.open(maskname).convert("RGB").resize(mask_size).getdata())
        elif self.masks[idx] + '_mask0.jpg' in os.listdir(self.maskpath):
            i = 0
            mask = np.zeros((mask_size[0],mask_size[1],3)).reshape([-1,3])
            while self.masks[idx] + '_mask' + str(i) + '.jpg' in os.listdir(self.maskpath):
                maskname = self.maskpath + '/' +  self.masks[idx] + '_mask' + str(i) + '.jpg'
                mask_ = np.array(Image.open(maskname).convert("RGB").resize(mask_size).getdata())
                mask = mask + mask_ # can add since masks should not overlap
                i += 1
        else:
            raise Exception("MISSING MASK for " + self.masks[idx])

        mask = mask.reshape((mask_size[0],mask_size[1],3))
        mask = mask.mean(axis=2)        
        mask = np.stack([mask,mask],axis=0)  # class 0 - embryo
        mask[1] = 255-mask[1]                # class 1 - nonembryo
        
        
        image = image.transpose([2,0,1])
        mask /= 255
        
        #original image sizes:
        #(1680, 1050, 3)
        #(640, 480, 3)
        return torch.Tensor(image), torch.Tensor(mask)

    def __len__(self):
        return len(self.data)


