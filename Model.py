'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ GG_1417 ]
# Author List:		[ Names of team members worked on this file separated by Comma: Anuj Verma, Sujal Ghadge, Pratik Magdum,Aman Patvegar ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 		
####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


##############################################################

################# ADD UTILITY FUNCTIONS HERE #################





##############################################################


def data_preprocessing(task_1a_dataframe):
    ''' 
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that 
    there are features in the csv file whose values are textual (eg: Industry, 
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function 
    should return encoded dataframe in which all the textual features are 
    numerically labeled.
    
    Input Arguments:
    ---
    `task_1a_dataframe`: [Dataframe]
                          Pandas dataframe read from the provided dataset 	
	
    Returns:
    ---
    `encoded_dataframe`: [Dataframe]
                          Pandas dataframe that has all the features mapped to 
                          numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''

    # Encode categorical features using LabelEncoder
    label_encoder = LabelEncoder()
    encoded_dataframe = task_1a_dataframe.copy()  # Make a copy to avoid modifying the original DataFrame

    # Impute missing numerical values using median
    # numerical_cols = encoded_dataframe.select_dtypes(include=np.number).columns
    # imputer = SimpleImputer(strategy='median')
    # encoded_dataframe[numerical_cols] = imputer.fit_transform(encoded_dataframe[numerical_cols])

    encoded_dataframe = task_1a_dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
    encoded_dataframe["Education"] = label_encoder.fit_transform(encoded_dataframe["Education"])
    encoded_dataframe["City"] = label_encoder.fit_transform(encoded_dataframe["City"])
    encoded_dataframe["Gender"] = label_encoder.fit_transform(encoded_dataframe["Gender"])
    encoded_dataframe["EverBenched"] = label_encoder.fit_transform(encoded_dataframe["EverBenched"])
    
    scaler = MinMaxScaler()
    encoded_dataframe[['JoiningYear', 'Age']] = scaler.fit_transform(encoded_dataframe[['JoiningYear', 'Age']])

    return encoded_dataframe



def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second 
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to 
                        numbers starting from zero
    
    Returns:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''

    # Extract features (all columns except the last one) and target (last column)
    features = encoded_dataframe.iloc[:, :-1]  # All columns except the last one
    target = encoded_dataframe.iloc[:, -1]  # Last column
    features_and_targets = [features, target]

    return features_and_targets

def load_as_tensors(features_and_targets):
    '''
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label
    
    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in 
                                                 batches, which are then fed into the model for processing
    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    # Extract features and targets
    features, targets = features_and_targets

    # Convert to PyTorch tensors
    # Ensure numerical features are in float32, and target in float32
    X_tensor = torch.tensor(features.values.astype(np.float32))
    y_tensor = torch.tensor(targets.values.astype(np.float32))  # Cast to float32

    # Split the data into training and validation sets (80% train, 20% validation)
    split_ratio = 0.8
    train_size = int(len(X_tensor) * split_ratio)
    X_train_tensor = X_tensor[:train_size]
    y_train_tensor = y_tensor[:train_size]
    X_test_tensor = X_tensor[train_size:]
    y_test_tensor = y_tensor[train_size:]

    # Create a TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Create a DataLoader for the training set
    batch_size = 32  # You can adjust this batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]

    return tensors_and_iterable_training_data



class Salary_Predictor(nn.Module):
    def __init__(self, input_features, hidden_layer1, hidden_layer2, hidden_layer3, output_classes):
        super(Salary_Predictor, self).__init__()

        # Define your layers here
        self.fc1 = nn.Linear(in_features=input_features, out_features=hidden_layer1)
        self.fc2 = nn.Linear(in_features=hidden_layer1, out_features=hidden_layer2)
        self.fc3 = nn.Linear(in_features=hidden_layer2, out_features=hidden_layer3)
        self.fc4 = nn.Linear(in_features=hidden_layer3, out_features=output_classes)

    def forward(self, x):
        # Apply activation functions and define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        predicted_output = torch.sigmoid(self.fc4(x))  # Apply sigmoid activation
        
        return predicted_output

    # @classmethod
    # def from_input_features(cls, features, hidden_layer1, hidden_layer2, hidden_layer3, output_classes):
    #     input_features = len(features.columns)
    #     return cls(input_features, hidden_layer1, hidden_layer2, hidden_layer3, output_classes)


def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.
    
    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    # Define and return a loss function (e.g., Mean Squared Error for regression)
    loss_function = nn.BCEWithLogitsLoss()  # Example: Mean Squared Error (MSE) loss
    
    return loss_function


def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible 
    for updating the parameters (weights and biases) in a way that 
    minimizes the loss function.
    
    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class
    
    Returns:
    ---
    `optimizer`: Pre-defined optimizer from PyTorch
    
    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    # Define and return an optimizer (e.g., Stochastic Gradient Descent - SGD)
    # Adjust the learning rate and other parameters as needed
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # Example: Stochastic Gradient Descent
    
    return optimizer


def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''
    number_of_epochs = 100
    return number_of_epochs


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer, patience=5):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model
    6. `patience`: Number of epochs with no improvement to wait for early stopping (default: 5)

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)
    '''    
    # Extract the training data
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader = tensors_and_iterable_training_data

    # Variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Train the model for the specified number of epochs
    for epoch in range(number_of_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Iterate through the training data in batches
        for batch_X, batch_y in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
    
            # Reshape batch_y to match the shape of outputs (binary classification)
            batch_y = batch_y.view(-1, 1)

            # Convert batch_y to float
            batch_y = batch_y.float()  # Convert to float
            
            # Calculate the loss
            loss = loss_function(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = loss_function(val_outputs, y_val_tensor.view(-1, 1)).item()

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping: No improvement for {patience} epochs. Stopping training.')
            break

    # Return the trained model
    return model



def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''    
    # Extract the validation data
    _, X_val_tensor, _, y_val_tensor, _ = tensors_and_iterable_training_data
    
    # Set the model to evaluation mode
    trained_model.eval()

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Predict on the validation set
        val_predictions = trained_model(X_val_tensor)
        _, predicted_labels = torch.max(val_predictions, 1)  # Choose the class with the maximum probability

    # Calculate accuracy
    correct_predictions = (predicted_labels == y_val_tensor).sum().item()
    total_predictions = len(y_val_tensor)
    model_accuracy = correct_predictions / total_predictions

    return model_accuracy

HIDDEN_LAYER1 = 128  # Number of neurons in the first hidden layer
HIDDEN_LAYER2 = 128  # Number of neurons in the second hidden layer
HIDDEN_LAYER3 = 128  # Number of neurons in the third hidden layer
OUTPUT_CLASSES = 1 
########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":
    # reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	

	# model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor(len(features_and_targets[0].columns), HIDDEN_LAYER1, HIDDEN_LAYER2, HIDDEN_LAYER3, OUTPUT_CLASSES)

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()  # Fixed indentation here

    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
                                       loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
