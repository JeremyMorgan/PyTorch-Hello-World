import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from simplenet import SimpleNet

if __name__ == '__main__':
    # Define the image transformations: convert images to tensors and normalize pixel values
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.30811,))])

    # Load the MNIST training and testing datasets with the defined transformations
    train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)

    # Create DataLoaders for the training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the input size, number of classes, learning rate, and number of epochs
    input_size = 28 * 28
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

    # Initialize the SimpleNet model, loss function (criterion), and optimizer
    model = SimpleNet(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Reshape the input data to match the expected input size of the model
            data = data.reshape(-1, input_size)
            # Calculate the scores (predictions) from the model
            scores = model(data)
            # Calculate the loss between the scores and the true targets
            loss = criterion(scores, targets)

            # Perform backpropagation and optimization
            optimizer.zero_grad()  # Reset gradients to zero before backpropagation
            loss.backward()  # Compute the gradients through backpropagation
            optimizer.step()  # Update the model's weights using the optimizer
