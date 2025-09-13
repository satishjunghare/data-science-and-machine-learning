# PyTorch Library

## Overview
PyTorch is an open-source machine learning framework developed by Meta (formerly Facebook) that provides a flexible and dynamic approach to building deep learning models. It's known for its Pythonic design, dynamic computational graphs, and excellent support for research and production. PyTorch has become one of the most popular deep learning frameworks due to its intuitive API and strong community support.

## Installation
```bash
# CPU version
pip install torch torchvision torchaudio

# CUDA version (for GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Latest version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

## Key Features
- **Dynamic Computational Graphs**: Define and modify networks on-the-fly
- **Pythonic Design**: Natural Python programming experience
- **GPU Acceleration**: Seamless CPU/GPU computation
- **Automatic Differentiation**: Built-in autograd for gradients
- **Rich Ecosystem**: Extensive libraries for vision, NLP, and more
- **Research Friendly**: Easy experimentation and prototyping
- **Production Ready**: TorchScript for deployment
- **Mobile Support**: PyTorch Mobile for edge devices

## Core Concepts

### Tensors
```python
import torch
import numpy as np

# Create tensors
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3, 4, 5])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Scalar shape: {scalar.shape}")
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {matrix.shape}")
print(f"3D Tensor shape: {tensor_3d.shape}")

# Tensor operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b  # Element-wise addition
d = torch.matmul(a.unsqueeze(1), b.unsqueeze(0))  # Matrix multiplication

# Move to GPU
if torch.cuda.is_available():
    a_gpu = a.cuda()
    print(f"Tensor on GPU: {a_gpu.device}")
```

### Automatic Differentiation (Autograd)
```python
# Create tensor with requires_grad=True
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

# Compute gradient
y.backward()
print(f"dy/dx = {x.grad}")

# Multiple variables
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.sum(x**2 + 2*x)
y.backward()
print(f"Gradients: {x.grad}")
```

## Building Neural Networks

### Sequential Model
```python
import torch.nn as nn
import torch.optim as optim

# Define neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model
model = SimpleNet()
print(model)
```

### Using nn.Sequential
```python
# Sequential model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)
```

## Training Models

### Basic Training Loop
```python
# Generate sample data
x = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

### Training with DataLoader
```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset and dataloader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training with batches
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/10], Average Loss: {total_loss/len(dataloader):.4f}')
```

## Convolutional Neural Networks (CNNs)

### CNN Architecture
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return torch.log_softmax(x, dim=1)

# Create CNN model
cnn_model = CNN()
```

### Training CNN
```python
# Prepare data for CNN
x_cnn = x.view(-1, 1, 28, 28)  # Reshape for CNN

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

cnn_model.train()
for epoch in range(5):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.view(-1, 1, 28, 28)  # Reshape for CNN
        
        outputs = cnn_model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/5], Average Loss: {total_loss/len(dataloader):.4f}')
```

## Recurrent Neural Networks (RNNs)

### LSTM Implementation
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Create LSTM model
lstm_model = LSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
```

## Transfer Learning

### Using Pre-trained Models
```python
import torchvision.models as models
import torchvision.transforms as transforms

# Load pre-trained ResNet
resnet = models.resnet50(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Modify the final layer
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Now only the final layer will be trained
for param in resnet.fc.parameters():
    param.requires_grad = True
```

## Data Loading and Preprocessing

### Custom Dataset
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Create dataset
dataset = CustomDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Data Transforms
```python
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Apply transforms
transformed_dataset = CustomDataset(x, y, transform=transform)
```

## Model Evaluation

### Evaluation Mode
```python
# Set model to evaluation mode
model.eval()

# Disable gradient computation
with torch.no_grad():
    test_outputs = model(x_test)
    _, predicted = torch.max(test_outputs.data, 1)
    
    # Calculate accuracy
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
```

### Model Saving and Loading
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model
torch.save(model, 'entire_model.pth')

# Load entire model
loaded_model = torch.load('entire_model.pth')
```

## GPU Acceleration

### Moving to GPU
```python
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move model to GPU
model = model.to(device)

# Move data to GPU
x = x.to(device)
y = y.to(device)

# Training on GPU
for epoch in range(10):
    outputs = model(x)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Advanced Features

### Custom Loss Functions
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # Custom loss computation
        loss = torch.mean((predictions - targets) ** 2)
        return loss

# Use custom loss
custom_criterion = CustomLoss()
```

### Learning Rate Scheduling
```python
# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training with scheduler
for epoch in range(10):
    # Training code...
    
    # Update learning rate
    scheduler.step()
    print(f'Learning rate: {scheduler.get_last_lr()[0]}')
```

### Gradient Clipping
```python
# Gradient clipping
max_grad_norm = 1.0

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
```

## TorchScript for Production

### Model Tracing
```python
# Trace the model
traced_model = torch.jit.trace(model, torch.randn(1, 784))

# Save traced model
traced_model.save('traced_model.pt')

# Load and use traced model
loaded_traced_model = torch.jit.load('traced_model.pt')
```

### Scripting
```python
# Script the model
scripted_model = torch.jit.script(model)

# Save scripted model
scripted_model.save('scripted_model.pt')
```

## Use Cases
- **Computer Vision**: Image classification, object detection, segmentation
- **Natural Language Processing**: Text classification, translation, generation
- **Speech Recognition**: Audio processing and speech-to-text
- **Recommendation Systems**: Collaborative filtering and content-based recommendations
- **Time Series Analysis**: Forecasting and anomaly detection
- **Reinforcement Learning**: Policy optimization and value function approximation
- **Generative Models**: GANs, VAEs, and other generative architectures

## Best Practices
1. **Data Preprocessing**: Normalize and augment data appropriately
2. **Model Architecture**: Start simple and gradually increase complexity
3. **Regularization**: Use dropout, batch normalization, and weight decay
4. **Learning Rate**: Use learning rate scheduling and monitoring
5. **Validation**: Always use validation data to monitor overfitting
6. **GPU Utilization**: Monitor GPU usage and optimize batch sizes
7. **Model Saving**: Save models regularly and use version control
8. **Memory Management**: Use gradient checkpointing for large models

## Advantages
- **Pythonic**: Natural Python programming experience
- **Dynamic**: Dynamic computational graphs for flexibility
- **Research Friendly**: Easy experimentation and prototyping
- **Community**: Large, active community and extensive documentation
- **Production Ready**: TorchScript for deployment
- **GPU Support**: Excellent GPU acceleration
- **Integration**: Works well with other Python libraries

## Limitations
- **Learning Curve**: Steep learning curve for beginners
- **Memory Usage**: Can be memory-intensive for large models
- **Debugging**: Complex debugging for custom operations
- **Mobile Deployment**: Limited compared to specialized frameworks
- **Ecosystem**: Smaller ecosystem compared to TensorFlow

## Related Libraries
- **TorchVision**: Computer vision utilities
- **TorchAudio**: Audio processing utilities
- **TorchText**: Text processing utilities
- **PyTorch Lightning**: High-level training framework
- **FastAI**: High-level API built on PyTorch
- **TensorFlow**: Alternative deep learning framework
- **NumPy**: Numerical computing foundation 