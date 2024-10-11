# Imports 

import torch
import numpy as np 
import torchvision
from torch import optim
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms, models
from collections import OrderedDict
import helper_functions

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


#Load Data 
data_dir = 'flowers'
test_dir = data_dir + '/test'

# Load the datasets with ImageFolder
image_datasets = helper_functions.load_datasets()
test_data = image_datasets['test']




# DataLoaders 
dataloaders = helper_functions.load_dataloaders()
testloader = dataloaders['test']

# from train import data_transforms['test'] 
# from train import image_datasets['test'] 
# from train import dataloaders['test'] 

load_model = helper_functions.load_checkpoint('checkpoint.pth')


def predict_or_not():
    
    answer= input("Would you like to make predictions?[y, n ]\n")
    helper_functions.vaildating_input(answer)
    if answer =='y':
        image_path= input("May you provide the directory of the image along with the image you would like to predict on\n")
      
        print("Working on it..........\n") 
     
        predict(image_path)
        # predict(image_path,load_model, class_dict=helper_functions.load_labels())
        
        print("Showing results........\n")
        sanity_check(image_path, load_model)
    
    
    elif answer =='n':
        print("Thanks for your time, Exiting.........") 

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Loading the image using PIL liberary 
    imageFile = torchvision.io.read_image(str(image_path)).type(torch.float32)/255.
   
    
    image_transform = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          # transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])



    processed_image = image_transform(imageFile)
 

    
    return processed_image
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
   
    ax.imshow(image)
    ax.axis(False)
    
  
   
    plt.show();
    
    
    return ax
# def predict(image_path, topk,  model=load_model, class_dict=helper_functions.load_labels(),  device= device ):
def predict(image_path, model=load_model, class_dict=helper_functions.load_labels(),  device= device, topk =5):
    """ 
    Pass an image path and let the model make predictions on that image and plot the image and the 5 top predictions 
    """
    model.to(device)

    # Load the image 
    image = process_image(image_path)
     
   

    
    # Making predictions 
    model.eval()

    with torch.inference_mode():
        img = image.unsqueeze(0)
        img = img.to(device) 
        logit = model(img) 
        ps = torch.exp(logit) 
 
    top_k_probs, top_k_class = ps.topk(topk, dim=1)
    top_k_probs = top_k_probs[0]
    top_k_class = top_k_class[0]
    
    
    top_class_titles=[]
    for i in range(len(top_k_class)):
        for k,v in class_dict.items():
            
            if v == top_k_class[i]:
                top_class_titles.append(k)
                



    return top_k_probs, top_class_titles
            


def sanity_check(image_path, model=load_model):
    # model=  load_checkpoint('checkpoint.pth')
    image = process_image(image_path)
    probs,classes = predict(image_path, load_model)
    probs = probs.cpu()

    fig = plt.figure(figsize =[9,9])
    plot1 = plt.subplot(4,4,2) 
    plot2 = plt.subplot(4,1,2)
     
    plot1.axis('off') 
    plot1.set_title(classes[0])
    plot1.imshow(image.permute(2,1,0)) 
   

    y_ticks = np.arange(5) 
    plot2.set_yticks(y_ticks) 
    plot2.set_yticklabels(classes)
    plot2.barh(y_ticks, probs) 
    plt.show() 
    
    print("\n")
    print("\n")
    print("The top most probable classes along with their probabilities are :\n")
    for i in range(len(classes)):
        
        print(f" Class {i+1} is : {classes[i]}  |  with probability {probs[i]}\n")



