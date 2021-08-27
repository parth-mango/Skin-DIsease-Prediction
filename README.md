# Skin-DIsease-Prediction
Predict differential diagnosis of different diseases for skin images.

This project was started after a friend of mine sent me an image of skin lesion and expected from me to suggest treatment for the disease. Being a general physician,
i had to consult with a dermatology specialist for the same and it struck my mind if AI can do such a task and to my surprise it performed quite well on a set of 
20+ skin diseases.  


### Dataset 
The data consist of 23 skin disease classes. The data was collected from http://www.dermnet.com/. It consist of a large set of images for different skin
diseases. The dataset is a good starting point to elucidate skin disease recognition.

### Repository

    train.py - Consists of:
               1. Dataloader Pipeline - Loading embeddings for the model 
               2. Adam Optimizer(considering to limitation of skin images per class)
               3. ReduceLRonPleatue - to achieve lower learning rates in later parts of training.
               4. nll loss as criterion
               5. training and testing functions
              
    model.py - Head of model - Uses the Resnet152 embeddings as inputs and finally outputs 23 class output 
             - involves fully connected layers
             
    
    embed.py - Resnet152 Embedder function
             - grabs the average pool embedding layer from ResNet152, which seems to do very nicely in organizing images.
             
    
    embeddings.py - Create embeddings of Resnet152 model - 32 batch groups - numpy objects of embeddings created.
    
    data.py  - Consists of:
               1. Dataloader Function - Load embeddings and targets as a dataloader object.
               2. Train test split function - 80% training sample and 20% test samples and reformatting of dataset into train and test folders
               3. Clone directory function - Defines the sub-directory structure and creates required directories without copying content.
    
    appstream.py - Streamlit implementation of the project. Here we create an app to allow users to upload images of skin lesions following which 
                   the model outputs the differential diagnosis for the condition presented in the image. We use pillow to load the given image and 
                   various streamlit functions to upload image, print predictions/outputs etc. Three differential diagnosis with their individual 
                   confidence score are printed for the user. 
    
    
    

 ![ Alt text](https://github.com/parth-mango/Skin-DIsease-Prediction/blob/main/demo/skin_dis.gif) 
 
 
 
    Ending Notes: This is a very simple implementation of the problem. I personally found it a nice usecase and it can be further improved and implemented
                  on a larger scale in peripheries where specialists are not available to diagnose such cases. 

                  For further discussions on Medical AI and its applications, you can reach out to me @ parth15237@gmail.com
