# Imports
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import argparse

# Taking user input using ArgumentParser
parse= argparse.ArgumentParser()
parse.add_argument('--path', type=str, default='flowers/test/16/image_06670.jpg')
parse.add_argument('--gpu', type=bool, default=False)
parse.add_argument('--top_k', type=int, default=3)
parse.add_argument('--checkpoint', type=str, default='checkpoint.pth')
parse.add_argument('--cat_json', type=str, default='cat_to_name.json')
arg= parse.parse_args()

# Loading json file
with open(arg.cat_json, 'r') as f:
    cat_to_name = json.load(f)

# Loading model
def load_model():
    checkpoint= torch.load('checkpoint.pth')
    
    if checkpoint['model_name']=='densenet':
        model= models.densenet121(pretrained=True)
        model_inp = 1024 # in_features for densenet
    elif checkpoint['model_name']=='vgg13':
        model= models.vgg13(pretrained=True)
        model_inp = model.classifier[0].in_features # auto determining in_features
    else:
        model= models.vgg16(pretrained=True)
        model_inp = model.classifier[0].in_features # auto determining in_features
    
    # Setting up layers
    classifier= nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(checkpoint['model_inp'], 1024)),
    ('relu1', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('hidden1', nn.Linear(1024, 512)),
    ('relu2', nn.ReLU()),
    ('hidden2', nn.Linear(512, checkpoint['hidd_units'])),
    ('fc2', nn.Linear(checkpoint['hidd_units'], 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier= classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['class_to_idx']
device= ('cuda' if arg.gpu else 'cpu')
load_model, class_to_idx= load_model()
idx_to_class= {m:n for n,m in class_to_idx.items()}
load_model.to(device);

# Processing image before predicting
def process_image(image):
    
    image= Image.open(image)
    size= 256, 256
    image.thumbnail(size)
    image= image.crop((16,16,240,240))
    
    np_image= np.array(image)
    np_image= np_image/255
    
    np_image[:,:,0]= (np_image[:,:,0]-0.485)/0.229
    np_image[:,:,1]= (np_image[:,:,1]-0.456)/0.224
    np_image[:,:,2]= (np_image[:,:,2]-0.406)/0.225
    np_image= np.transpose(np_image, (2, 0, 1))
    return np_image

# Predicting image
def predict(image_path, model, topk=arg.top_k):
    
    model.eval()
    if device=='cuda':
        image= torch.cuda.FloatTensor([process_image(image_path)])
    else:
        image= torch.FloatTensor([process_image(image_path)])
    output= model.forward(image)
    prob= torch.exp(output).data.cpu().numpy()[0]
    
    topk_idx= np.argsort(prob)[-topk:][::-1]
    topk_class= [idx_to_class[c] for c in topk_idx]
    topk_prob= prob[topk_idx]
    return topk_prob*100, topk_class

img_loc= arg.path
img_file= Image.open(img_loc)

probs, classes= predict(img_loc, load_model)

# Printing out results
for p, c in zip(probs, classes):
    print(cat_to_name[c], ': ', float('{0:.2f}'.format(p)), '%')