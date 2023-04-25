import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from simplenet import SimpleNet
import matplotlib.pyplot as plt

# Data loading: define image transformations, load the MNIST test dataset, and create a DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained model
input_size = 28 * 28
num_classes = 10

model = SimpleNet(input_size, num_classes)
model.load_state_dict(torch.load('trained_model.pth'))

# Evaluation function: calculate the accuracy of the model on a given dataset
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(-1, input_size)
            scores = model(x)
            _, predictions = torch.max(scores, 1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# Print the accuracy on the test dataset
print(f"Accuracy on test dataset: {check_accuracy(test_loader, model):.2f}")

# Visualization function: display sample images, their true labels, and the model's predictions
def visualize_predictions(loader, model, num_samples=10):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.reshape(-1, input_size)
            scores = model(x)
            _, predictions = torch.max(scores, 1)

            # Plot sample image, true label, and predicted label
            axes[idx].imshow(x[idx].reshape(28, 28).numpy(), cmap='gray')
            axes[idx].set_title(f"True: {y[idx].item()}, Pred: {predictions[idx].item()}")
            axes[idx].axis("off")

            if idx == num_samples - 1:
                break

    plt.show()
    model.train()

# Visualize predictions on the test dataset
visualize_predictions(test_loader, model, num_samples=10)
