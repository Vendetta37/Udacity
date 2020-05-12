import torch
import numpy as np
import argparse
from model import load_checkpoint, cuda_available
from process import get_input_args, process_image, predict, category_names

def main():
    in_arg = get_input_args()
    
    # Use CUDA if available
    device = cuda_available(in_arg.gpu)
    
    # Load checkpoint
    if in_arg.load_checkpoint == 'None':
        print('Please choose a input checkpoint')
    
    else:
        model, epochs_start = load_checkpoint(in_arg.load_checkpoint)
        print('Checkpoint successfully loaded..')
    
    # Process a PIL image 
    if in_arg.input_image == 'None':
        print('Please choose a input image')
    
    else:
        adj_image = process_image(in_arg.input_image)
        print('Image successfully loaded..')
        
    # Predict the class from an image file     
    adj_image, probs, classes = predict(in_arg.input_image, model, adj_image, in_arg.topk, device)
    
    # Mapping from category label to category name
    cat_to_name = category_names(in_arg.category_names)    
    actual_classes = [cat_to_name[n] for n in classes]
    
    for actual_class, prob in zip(actual_classes, probs):
        print(f'{actual_class}: {prob*100:.3f}%')
    
if __name__ == '__main__':
    main()