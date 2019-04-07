#!/usr/bin/env python3
#importing libraries

import argparse
from ops import Ops, Network
import torch
#from PIL import Image


def get_args():
    #defining the parser
    parser = argparse.ArgumentParser(description='training arguments')
    parser.add_argument('data_dir', type=str, default="none", help="directory of the training set")
    parser.add_argument('-s','--save_dir', type=str, default="checkpoint", help='the directory to save the checkpoint')
    parser.add_argument('-a','--arch', type=str, default="vgg16", help='the architecture used eg. vgg13, vgg16')
    parser.add_argument('-l','--learning_rate', type=float, default=0.001, help='the learning rate')
    parser.add_argument('-u','--hidden_units', type=int, default=512, help='number of nodes in the hidden layer')
    parser.add_argument('-i','--in_units', type=int, default=25088, help='number of input nodes in the input layer')
    parser.add_argument('-o','--out_units', type=int, default=102, help='number of output nodes in the output layer')
    parser.add_argument('-e','--epochs', type=int, default=10, help='number of nodes in the hidden layer')
    parser.add_argument('--gpu',help='the architecture used eg. vgg13, vgg16', action='store_true')
    return parser


def main():
    args = get_args().parse_args()
    print('dataset dir: ', args.data_dir)
    print('save dir: ', args.save_dir)
    print('Architecture: ', args.arch)
    print('Learning rate: ', args.learning_rate)
    print('Hidden units: ', args.hidden_units)
    print('EPOCHS: ', args.epochs)
    print('GPU: ', args.gpu)
    
    #getting the directories of training and validation sets
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    
    #prepping training and validation sets
    trainloader, class_to_idx = Ops.make_trainingloader(train_dir)
    validationloader = Ops.make_validationloader(valid_dir)
    #constructing the model 
    model = Ops.make_network(args.in_units, args.out_units, args.hidden_units, args.arch, args.learning_rate)
    #training the network model
    model = Ops.train_model(model, args.epochs, args.gpu, args.learning_rate, trainloader, validationloader)
    #creating checkpoint
    checkpoint = {'input_size': args.in_units,
              'output_size': args.out_units,
              'hidden_layers': [each.out_features for each in model.classifier.layers],
              'drop':args.learning_rate,
              'state_dict': model.classifier.state_dict(),
              'class_to_idx': class_to_idx}
    torch.save(checkpoint, args.save_dir + '.pth')

if __name__ == "__main__":
    main()
