import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
import torchvision
from torchvision import datasets, transforms, models# Imports here
import PIL
from PIL import Image
import glob, os
import random
import numpy as np
import torchvision.models as models
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('image_path',
                    help='path to singular image')
parser.add_argument('checkpoint',
                    action='store',
                    default='my_checkpoint.pth',
                    help='specified path to load model')
parser.add_argument('--top_k',
                    action='store',
                    dest='topk',
                    default=5,
                    type=int,
                    help='specified top k values')
parser.add_argument('--category_names',
                    action='store',
                    default='cat_to_name.json',
                    help='Category names for images')
parser.add_argument('--gpu',
                    action='store_true',
                    dest='cuda',
                    help='specifies using gpu for inference')
                    
                    

result = parser.parse_args()

checkpoint = torch.load(result.checkpoint)
#print('keys: {}'.format(checkpoint))

model = checkpoint['model']
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])


cat_to_name = None
with open(result.category_names, 'r') as f:
    cat_to_name = json.load(f)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image)
    img = img.resize((256, 256))
    center_crop = 0.5*(256-224)
    img = img.crop((center_crop, center_crop, 256 - center_crop, 256 - center_crop ))
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    return img.transpose(2, 0, 1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and result.cuda else 'cpu')
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(np.array(np.array([image]))).float()
    image = image.to("cuda")
    if device == "cuda":
        image = image.cuda()
        
    output = model.forward(image)
    probabilities = torch.exp(output).data
    
    probs = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
                           ind.append(list(model.class_to_idx.items())[i][0])
                       
    classes = []
    for i in range(5):
        classes.append(ind[index[i]])
                       
    return probs, classes

image_random = random.choice(os.listdir(result.image_path))
img_path = result.image_path + image_random
probs, classes = predict(img_path, model, result.topk)
max_index = np.argmax(probs)
final_probability = probs[max_index]
label = classes[max_index]



print('flower name: {}, probability: {}'.format(cat_to_name[label], final_probability))
