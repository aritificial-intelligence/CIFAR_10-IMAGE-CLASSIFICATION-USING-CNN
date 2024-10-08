from resnet20_cifar import resnet20
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image
from thop import profile
import time

# Set random seed for reproducibility
seed_value = 30
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Updated normalization
])

# Load CIFAR-10 dataset
def load_data(batch_size=64):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader




# Training function
def train_model(model, num_epochs=10, batch_size=64):
    train_loader, test_loader = load_data(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # To store accuracy for all epochs
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        # Training
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

        train_accuracy = correct_train_predictions / total_train_samples
        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracies.append(train_accuracy)
    # Adjust learning rate after specific epochs
        if epoch == 5 or epoch == 8:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10
        # Testing
        model.eval()
        running_test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test_samples += labels.size(0)
                correct_test_predictions += (predicted == labels).sum().item()

        test_accuracy = correct_test_predictions / total_test_samples
        avg_test_loss = running_test_loss / len(test_loader)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        

    # Calculate and print mean and std of train and test accuracies
    train_mean = np.mean(train_accuracies)
    train_std = np.std(train_accuracies)
    test_mean = np.mean(test_accuracies)
    test_std = np.std(test_accuracies)

    print(f'\nTrain Accuracy: Mean = {train_mean:.4f}, Std = {train_std:.4f}')
    print(f'Test Accuracy: Mean = {test_mean:.4f}, Std = {test_std:.4f}')
    
    # Calculate MACs and parameters
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Create a dummy input tensor
    macs, params = profile(model, inputs=(dummy_input,))  # Pass the dummy input as a single-element tuple
    print(f"SimpleCNN: MACs = {macs / 1e6:.2f} M, Parameters = {params / 1e6:.2f} M")

    # Save the trained model after training
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/cnn_model.pth')
    print('Model saved to ./model/cnn_model.pth')


# Testing function for custom images
def test_image(model, image_path):
    model.eval()

    # Load the image and transform it
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    class_idx = predicted.item()

    print(f"Predicted class index: {class_idx}, Class name: {CIFAR10_CLASSES[class_idx]}")

    # Visualize the first conv layer
    first_conv_output = model.conv1(image).detach().cpu()
    visualize_conv_layer(first_conv_output)

# Visualize convolutional layer
def visualize_conv_layer(conv_output, save_path='CONV_rslt.png'):
    conv_output = conv_output.squeeze(0)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < conv_output.size(0):
            ax.imshow(conv_output[i].numpy(), cmap='gray')
        ax.axis('off')
    plt.savefig(save_path)
    print(f"Convolutional layer visualization saved to {save_path}")

# Test accuracy of pre-trained ResNet-20
def test_resnet20():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained ResNet-20 model
    model = resnet20()
    model.load_state_dict(torch.load("resnet20_cifar10_pretrained.pt", map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare the CIFAR-10 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Updated normalization for ResNet-20
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    # Evaluate the model
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the CIFAR-10 test dataset: {accuracy:.2f}%')
   # Count parameters and MACs for ResNet-20
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Create a dummy input tensor
    macs, params = profile(model, inputs=(dummy_input,))  # Pass the dummy input as a single-element tuple
    print(f"ResNet-20: MACs = {macs / 1e6:.2f} M, Parameters = {params / 1e6:.2f} M")
# Inference speed test function


def inference_speed_test(model, num_iterations=1000):
    model.eval()

    image_path = './dog_small.png'  # Replace with your image path
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension
    input_tensor = image

    # Warm-up iterations
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)

    end_time = time.time()

    # Calculate average inference time
    total_time = end_time - start_time
    average_time = total_time / num_iterations

    print(f"Average inference time for input: {average_time:.6f} seconds")



# Main function to handle different commands
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("python CNNclassify.py train")
        print("python CNNclassify.py test <image_path>")
        print("python CNNclassify.py resnet20")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'train':
        model = SimpleCNN().to(device)
        train_model(model)

    elif command == 'test':
        if len(sys.argv) < 3:
            print("Please provide the path to the image to test.")
            sys.exit(1)
        image_path = sys.argv[2]
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load('./model/cnn_model.pth'),strict=False)
        test_image(model, image_path)

    elif command == 'resnet20':
       
        test_resnet20()
    elif command == 'speed_test':
        if len(sys.argv) < 3:
            print("Please provide the model type (simple or resnet) and input type (dummy or real).")
            sys.exit(1)
        model_type = sys.argv[2]

        if model_type == 'simple':
            model = SimpleCNN().to(device)
            model.load_state_dict(torch.load('./model/cnn_model.pth'), strict=False)
            inference_speed_test(model)
        elif model_type == 'resnet':
            model = resnet20()
            model.load_state_dict(torch.load("resnet20_cifar10_pretrained.pt", map_location=device))
            model = model.to(device)
            inference_speed_test(model)
    else:
        print("Invalid command.")
