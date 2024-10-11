# Imports 
import os
import json
import torch
import argparse
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict


# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_data_dir():
    """ Loading and setting directory 
    Returns : a dictionary with associated directories 
    """
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_dirs = {'train': train_dir , 'valid': valid_dir, 'test': test_dir}
   
    return data_dirs



    
########## Data Transformations 

train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                       transforms.RandomResizedCrop(200),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ColorJitter(brightness=0.4, contrast= 0.3,saturation=0.2 , hue=0.1 ),
                                       transforms.CenterCrop(200),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(200),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(200),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    
     
data_transforms =  {'train': train_transforms, 'valid': valid_transforms, 'test': test_transforms}

  



    
def load_datasets():
   

    data_dirs = load_data_dir()
    
    train_data = datasets.ImageFolder(data_dirs['train'], transform = data_transforms ['train'])
    valid_data = datasets.ImageFolder(data_dirs['valid'], transform = data_transforms['valid']) 
    test_data = datasets.ImageFolder(data_dirs['test'], transform = data_transforms['test'])

    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
    return image_datasets


def load_dataloaders():
    image_datasets = load_datasets()
    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32 )

    dataloaders = {'train': trainloader, 'valid': validloader, 'test': testloader}
    return dataloaders
    
  
 
# These functions to map class numbers from model.class_to_idx to its proper names from jason file  
def get_category_names():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def load_labels():
    """ helpere function to extract the proper category name of flowers matching their class names in numbers  """
    image_datasets = load_datasets() 
    class_to_idx = image_datasets['train'].class_to_idx
    cat_to_name = get_category_names() 
    labels_dict= {}
    # for k, v in model.class_to_idx.items():
    for k, v in class_to_idx.items():
        for k_1, v_1 in cat_to_name.items(): 
            if k == k_1:
                labels_dict[v_1] = v
    
    return labels_dict



def get_input():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arch', type = str, default = 'densenet', choices=['densenet','alexnet','vgg'], 
                    help = 'Model Architectures')
    parser.add_argument('--lr', type = float, default = 0.001, 
                    help = 'Learning Rate')
    parser.add_argument('--h_units', type = int, default = 1024, 
                    help = 'Number of hidden units')
                    
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'Number of Epochs')
                    
    parser.add_argument('--disable_gpu', action= 'store_true', 
                    help = 'Disabling GPU')
    in_args = parser.parse_args()
    
    
    
    assert in_args.epochs > 0, " invalid value for number of epochs" 
    
    return in_args 
                     

in_model=get_input().arch.lower()






def classifier( in_model='densenet', dropout = 0.5, hidden_units = 1024):
    """ This function has defaulted values in case the user has not picked something sothat it would not crash
    it will take the user input and search for the model that matched the user selection and based on the preprocess
    
    Args : user selection of a model name from 3 sugeestions 
    
    Returns :  A model based on what was provided by the user or default vgg model after fitting it to the data 
    """
    if in_model == 'densenet':
       model = models.densenet121(pretrained= True) # 1024 
       in_feautes = 1024
       
    
    elif in_model == 'alexnet':
        model = models.alexnet(pretrained= True) # 9216 
        in_feautes = 9216 
     
    elif in_model == 'vgg':
        model =  models.vgg19(pretrained= True) # input is 25088
        in_feautes = 25088

    for param in model.parameters():
        param.requires_grad = False
    # replicate the classifier of the pretrained model to fit our data     
    classifier = nn.Sequential(OrderedDict([ 
                           ("fc1", nn.Linear(in_feautes, hidden_units)),
                           
                           ("relu", nn.ReLU()),
                           
                           ("dropout", nn.Dropout(dropout)),
                           
                           ("fc2", nn.Linear(hidden_units, 102)),   # our out_features will be the same no matter what model we pick to represent the 102 classes of flowers 
                          
                       
                          ("output", nn.LogSoftmax(dim=1))]))
                     


    model.classifier = classifier
    return model

    

  
  
  
  
def training_loop (model, train_dataloader,valid_dataloader, criterion, optimizer, device=device):
    
    model.to(device) 
    model.train()
    train_loss = 0    # track our training loss
    train_accuracy = 0
    steps = 0
    print_every = 26 

    #loop through our data 
    for X, y in train_dataloader:
        #make sure that we move our data to the GPU if it is available
        X = X.to(device)
        y = y.to(device)

        steps+=1 

        #get the log probabilities
        y_pred = model(X)

        #use the log probabilities to get our loss 
        loss = criterion(y_pred, y)
        
        #keep track of our training loss
        train_loss += loss.item()
        train_accuracy += torch.mean(torch.eq(y_pred.argmax(dim=1), y).type(torch.FloatTensor))

        #  zero the gradient
        optimizer.zero_grad()

        # backpropagation 
        loss.backward()

        # gardient descent 
        optimizer.step()
   

        if steps % print_every == 0:
            
            train_loss /= print_every
            train_accuracy /= len(train_dataloader)
            model.eval()
            v_loss = 0
            v_accuracy = 0
            with torch.no_grad():
                
                for X, y in valid_dataloader:    
                    #Setting data to the target device
                    X = X.to(device) 
                    y = y.to(device) 
    
                    # Do forward pass
                    y_pred = model(X)
                 
                    #Calculate the loss
                    loss = criterion(y_pred, y)
                    v_loss += loss.item()
    
                    v_accuracy += torch.mean(torch.eq(y_pred.argmax(dim=1), y).type(torch.FloatTensor))
                v_loss/= len(valid_dataloader)
                v_accuracy /= len(valid_dataloader)
                v_accuracy*=100
            print(f"Train Loss : {train_loss : .5f} | Train Accuracy: { train_accuracy : .2f}% | Validation Loss : {v_loss : .5f} | Valiadtion Accuracy: {v_accuracy : .2f}%\n")








def testing_model(model, data_loader, criterion, device=device):


  
  """ Returns a dictionary containing the results of model predicting on our dataloader of our test set """

  loss = 0
  accuracy = 0 
  model.to(device)

  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:
        
        #set them to the device
        X = X.to(device) 
        y = y.to(device)
        
        # Make predictions 
        y_pred = model(X)

        #calculate the loss
        loss = criterion(y_pred, y)
        loss += loss.item()

        accuracy += torch.mean(torch.eq(y_pred.argmax(dim=1), y).type(torch.FloatTensor))

            

                                                        
                             
                            

    loss/= len (data_loader)
    accuracy /= len(data_loader) 
    accuracy *= 100 
    
    



  #Return the results as a dictionary
  return {"model_name" :type(model).__name__ , # only works when model was created with a class
          "model_loss": loss.item(), 
          "model_accuracy" : accuracy}

                         



      
# Helping function to validate user input      
def vaildating_input(answer):
    # answer.lower()
    answer = answer.lower()
    if answer == 'y' or answer =='n':
        print(f" Your choice is {answer}, and working on it\n")
        
        return answer
    else:
        print(f"Your choice is {answer} and does not match the provided options\n")
    
 
                         
                         
                              
            
        
    
# loads a checkpoint and rebuilds the model
def load_checkpoint(file_path):
    
    checkpoint = torch.load(file_path)
    

    model_name = checkpoint['model_name']
    state_dict = checkpoint['state_dict']
    epochs= checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    optimizer= checkpoint['optimizer'] 
    classifier = checkpoint['classifier']
    
    if model_name =='DenseNet':
        model = models.densenet121(pretrained= True)
        
    elif model_name == 'VGG':

        model = models.vgg19(pretrained = True)
        
    elif model_name =='AlexNet':
        model = models.alexnet(pretrained= True)

  

    for param in model.parameters():
        param.requires_grad = False

    model.classifier= classifier 
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)
   
    

    return model
        

    

