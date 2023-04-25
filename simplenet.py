import torch
import torch.nn as nn

# Define a simple neural network class called SimpleNet, which inherits from the PyTorch nn.Module class
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        # Call the constructor of the parent class (nn.Module)
        super(SimpleNet, self).__init__()
        
        # Define the first fully connected (Linear) layer with input_size number of input features and 50 hidden units
        self.fc1 = nn.Linear(input_size, 50)
        
        # Define the second fully connected (Linear) layer with 50 input features and num_classes number of output features
        self.fc2 = nn.Linear(50, num_classes)

    # Define the forward pass of the neural network
    def forward(self, x):
        # Pass the input through the first fully connected layer and apply the ReLU activation function
        x = torch.relu(self.fc1(x))
        
        # Pass the output of the first layer through the second fully connected layer
        x = self.fc2(x)
        
        # Return the final output of the neural network
        return x
