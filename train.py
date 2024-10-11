# Imports 

import torch
import numpy as np 
from torch import nn
from torch import optim
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms, models
from collections import OrderedDict
import helper_functions 
import predict

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#Hyperparameters 
print("\n")
print("Training the model starts on the following choices.........................................\n")
print("The model architecture: " ,helper_functions.get_input().arch)
print("The Learning rate: " , helper_functions.get_input().lr)
print("The number of hidden units : " , helper_functions.get_input().h_units)
print("The number of epochs : " , helper_functions.get_input().epochs)
print("GPU is:" ,'disabled' if helper_functions.get_input().disable_gpu else 'enabled')

model_arch = helper_functions.get_input().arch.lower()
lr = helper_functions.get_input().lr 
hidden_layers = helper_functions.get_input().h_units
epochs = helper_functions.get_input().epochs



# Load the datasets with ImageFolder
image_datasets = helper_functions.load_datasets()

train_data = image_datasets['train']
valid_data = image_datasets['valid']
test_data = image_datasets['test']






# DataLoaders 
dataloaders = helper_functions.load_dataloaders()

trainloader = dataloaders['train']
validloader = dataloaders['valid']
testloader = dataloaders['test']







    
model = helper_functions.classifier(in_model=model_arch,  hidden_units =hidden_layers)
# Define loss function , "Negative Log Likelihood Loss"  
criterion = nn.NLLLoss()


# Define an optimizer " Adam optimizer with weight decay" 
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)






################################## Training and validating 

print("\n")
# print("Training the model starts.........................................")
print("\n")
for epoch in range(epochs):
    print(f"Epoch : {epoch+1}/{epochs}\n  -----------------------")
    # # Training Loop
    helper_functions.training_loop(model, dataloaders['train'],dataloaders['valid'], criterion, optimizer,device)
    







######################## Testing model predictions 
print("\n")
print("Testing model results......................\n") 

print(helper_functions.testing_model(model, dataloaders['test'], criterion, device))





# function to save model  
def save_model():
  
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'model_name': type(model).__name__,
              'classifier': model.classifier,
              'epochs':epochs,
              'optimizer':optimizer.state_dict(), 
              'state_dict':model.state_dict(),
              'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    
    
    
    
    
# Saving model     
print("\n")
saving_model = input("Do you want to save the model? [y, n]\n") 
helper_functions.vaildating_input(saving_model)
if saving_model =='y':
   
    save_model()
    print(" Model successfully saved\n") 
else:
    print("Model is not saved\n")  
    

    
# Linking to the prediction stage to make prediction if the user wants     
predict.predict_or_not()

