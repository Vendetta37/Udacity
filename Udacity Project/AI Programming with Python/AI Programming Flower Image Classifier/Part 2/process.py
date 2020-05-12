import torch
import numpy as np
import json
import argparse
from PIL import Image

def get_input_args():
    
    #Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=int, default=1,
                        help='use CUDA if GPU is available')
    
    parser.add_argument('--load_checkpoint', type=str, default='None',
                        help='input checkpoint to load')
    
    parser.add_argument('--input_image', type=str, default='None',
                        help='Input a single image')
    
    parser.add_argument('--topk', type=int, default=5,
                        help='Select the number of most likely classes')
    
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Load a JSON file for mappings')
    
    return parser.parse_args()

# Process a PIL image
def process_image(image): 
    im = Image.open(image)
    
    if im.size[0] >= im.size[1]:
        ratio = im.size[0] // im.size[1]
        size = 256 * ratio, 256
        
    elif im.size[0] < im.size[1]:
        ratio = im.size[1] // im.size[0]
        size = 256, 256 * ratio
    
    im = im.resize(size, Image.ANTIALIAS)
    
    left = (im.size[0] - 224)/2
    top = (im.size[1] - 224)/2
    right = (im.size[0] + 224)/2
    bottom = (im.size[1] + 224)/2
    im = im.crop((left, top, right, bottom))
    
    
    np_image = np.array(im)
    np_image = np_image / 255
    means = np.array([0.485, 0.456, 0.406])
    sds = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - means)/sds
    
    adj_image = normalized_image.transpose(2, 0, 1)
    
    return adj_image

# Predict the class from an image file 
def predict(image_path, model, adj_image, topk, device): 
    model.to(device);
    
    image = torch.from_numpy(adj_image).type(torch.FloatTensor).to(device)
    image.unsqueeze_(0)   
        
    ps = torch.exp(model(image))
    probs, classes = ps.topk(topk, dim=1)
    probs, classes = probs.to('cpu'), classes.to('cpu')

    probs = probs.detach().numpy().tolist()[0]
    classes = classes.detach().numpy().tolist()[0]
    idx_to_class = {val:key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[c] for c in classes]
    
    return adj_image, probs, classes

# Mapping from category label to category name
def category_names(map_file):
    
    with open(map_file, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name