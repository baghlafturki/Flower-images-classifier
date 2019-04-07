#!/usr/bin/env python3
from PIL import Image  
from ops import Ops, Network
import torch.nn.functional as F
import torch
import json
import argparse
 

def get_args():
    #defining the parser
    parser = argparse.ArgumentParser(description='training argument')
    parser.add_argument('img_dir', type=str, help="path of the image")
    parser.add_argument('checkpoint', type=str, help="the checkpoint of the network")
    parser.add_argument('-k','--top_k', type=int, default = 3, help='number of classes with the highest probability')
    parser.add_argument('-a','--category_names', type=str, default="cat_to_name.json", help='path to the file with class names')
    parser.add_argument('--gpu',help='use GPU to do infrences', action='store_true')    
    return parser
def main():
    args = get_args().parse_args()
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(args.checkpoint)
    model = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                            checkpoint['drop'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    image = Ops.process_image(args.img_dir)
    #reading the class to name file 
    cat_to_name = json.load(open(args.category_names,'r'))
    names, probs, index = [], [], []
    img = torch.Tensor(image)
    img = img.resize_(1,25088)
    if args.gpu:
        img = img.cuda()
    model.eval()
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    with torch.no_grad():
        result = F.softmax(model.forward(img), dim=1).topk(args.top_k)
        
        
        for i in result[1][0]:
            index.append(i.item())
    top_classes = [index_to_class[each] for each in index]
  
    for i in range(len(top_classes)):
        print("(" + str(round(float(result[0][0].data[i]),2)) + ") " + cat_to_name[top_classes[i]])
    
    
    
    
    
if __name__ == "__main__":
    main()
