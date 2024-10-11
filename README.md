In this project, I developed a Python application that trains a model on a dataset, allowing it to recognize and classify images effectively. Here are some key highlights:

🔍 Data Augmentation and normalization: to increase the size of the training set and to help to the model to be robust with the help of torchvision transforms, where normalization helps improve the gradients of the outputs of the non-linear functions. 


🧠 Model Training: Utilizing a pretrained model, loaded in pre-trained weights from a network trained on a large dataset, freezed all the weights in the lower (convolutional) layers: the layers to freeze are adjusted depending on similarity of new task to original dataset. Replaced the upper layers of the network with a custom classifier. Finally, trained only the custom classifier layers for the task thereby optimizing the model for smaller dataset

🖼️ Prediction Capability: After training, I implemented functionality to predict new images using the trained model, showcasing its real-world applicability.
