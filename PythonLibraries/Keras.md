# Keras

## Overview

**Keras** is a high-level neural network API that was originally developed as an independent library and is now the official high-level API for TensorFlow. It provides a user-friendly interface for building and training deep learning models with minimal code complexity.

## Installation

```bash
# Install Keras (comes with TensorFlow)
pip install tensorflow

# Or install standalone Keras (for backend flexibility)
pip install keras
```

## Key Features

- **User-friendly API**: Simple and intuitive interface for building neural networks
- **Multiple backends**: Supports TensorFlow, Theano, and CNTK (though TensorFlow is now the primary backend)
- **Modular design**: Easy to combine different layers and components
- **Built-in optimizers**: Comprehensive set of optimization algorithms
- **Callbacks system**: Flexible training monitoring and control
- **Model serialization**: Easy saving and loading of models
- **Pre-trained models**: Access to popular architectures (VGG, ResNet, etc.)

## Core Concepts

### 1. Sequential Model
The simplest way to build a neural network by stacking layers sequentially.

```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()
```

### 2. Functional API
More flexible way to build complex models with multiple inputs/outputs.

```python
from tensorflow import keras
from tensorflow.keras import layers

# Define inputs
input_layer = layers.Input(shape=(784,))
hidden_1 = layers.Dense(128, activation='relu')(input_layer)
dropout_1 = layers.Dropout(0.2)(hidden_1)
hidden_2 = layers.Dense(64, activation='relu')(dropout_1)
dropout_2 = layers.Dropout(0.2)(hidden_2)
output_layer = layers.Dense(10, activation='softmax')(dropout_2)

# Create model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. Model Subclassing
Most flexible approach for custom model architectures.

```python
from tensorflow import keras
from tensorflow.keras import layers

class CustomModel(keras.Model):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

# Create and compile model
model = CustomModel()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Common Layer Types

### Dense Layers
```python
# Fully connected layer
dense_layer = layers.Dense(
    units=64,                    # Number of neurons
    activation='relu',           # Activation function
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=keras.regularizers.l2(0.01)
)
```

### Convolutional Layers
```python
# 2D Convolution
conv2d = layers.Conv2D(
    filters=32,                  # Number of filters
    kernel_size=(3, 3),          # Filter size
    strides=(1, 1),              # Stride
    padding='same',              # Padding type
    activation='relu'
)

# 1D Convolution (for sequences)
conv1d = layers.Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu'
)
```

### Recurrent Layers
```python
# LSTM layer
lstm_layer = layers.LSTM(
    units=128,                   # Number of units
    return_sequences=True,       # Return full sequence
    dropout=0.2,                 # Dropout rate
    recurrent_dropout=0.2        # Recurrent dropout
)

# GRU layer
gru_layer = layers.GRU(
    units=64,
    return_sequences=False,
    dropout=0.1
)
```

### Pooling and Normalization
```python
# Max pooling
max_pool = layers.MaxPooling2D(pool_size=(2, 2))

# Average pooling
avg_pool = layers.AveragePooling2D(pool_size=(2, 2))

# Batch normalization
batch_norm = layers.BatchNormalization()

# Layer normalization
layer_norm = layers.LayerNormalization()
```

## Training and Evaluation

### Basic Training
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Generate sample data
X_train = np.random.random((1000, 784))
y_train = np.random.randint(0, 10, (1000,))

X_test = np.random.random((200, 784))
y_test = np.random.randint(0, 10, (200,))

# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Training with Callbacks
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Train with callbacks
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
```

### Custom Training Loop
```python
import tensorflow as tf

# Custom training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(y, predictions)

# Custom validation step
@tf.function
def test_step(x, y):
    predictions = model(x, training=False)
    t_loss = loss_object(y, predictions)
    
    test_loss(t_loss)
    test_accuracy(y, predictions)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    for x_batch, y_batch in test_dataset:
        test_step(x_batch, y_batch)
    
    print(f'Epoch {epoch+1}:')
    print(f'  Loss: {train_loss.result():.4f}')
    print(f'  Accuracy: {train_accuracy.result():.4f}')
    print(f'  Val Loss: {test_loss.result():.4f}')
    print(f'  Val Accuracy: {test_accuracy.result():.4f}')
```

## Pre-trained Models

### Image Classification Models
```python
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

# Load pre-trained model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Transfer Learning
```python
# Fine-tuning approach
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze first layers
    layer.trainable = False

# Use lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Data Preprocessing

### Image Preprocessing
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess data
train_generator = datagen.flow_from_directory(
    'train_data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train with generator
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10
)
```

### Text Preprocessing
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

# Convert to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=100)

# Create embedding layer
embedding_layer = layers.Embedding(
    input_dim=10000,
    output_dim=128,
    input_length=100
)
```

## Model Saving and Loading

### Save and Load Models
```python
# Save entire model
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save model architecture only
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)

# Load model architecture
with open('model_architecture.json', 'r') as f:
    model_json = f.read()
loaded_model = keras.models.model_from_json(model_json)

# Save weights only
model.save_weights('model_weights.h5')

# Load weights
model.load_weights('model_weights.h5')
```

## Use Cases

### 1. Image Classification
```python
# CNN for image classification
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 2. Text Classification
```python
# LSTM for text classification
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=100),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

### 3. Time Series Forecasting
```python
# LSTM for time series
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(1)
])
```

## Best Practices

### 1. Model Architecture
- Start simple and gradually increase complexity
- Use appropriate activation functions (ReLU for hidden layers, softmax/sigmoid for output)
- Apply dropout to prevent overfitting
- Use batch normalization for faster training

### 2. Training
- Use appropriate learning rates (start with 0.001 for Adam)
- Implement early stopping to prevent overfitting
- Use data augmentation for small datasets
- Monitor both training and validation metrics

### 3. Data Preprocessing
- Normalize/standardize input data
- Handle missing values appropriately
- Use appropriate loss functions for your task
- Balance your dataset if dealing with imbalanced classes

### 4. Performance Optimization
- Use GPU acceleration when available
- Batch your data appropriately
- Use mixed precision training for faster training
- Profile your model for bottlenecks

## Advantages

1. **User-friendly**: Simple and intuitive API
2. **Flexible**: Multiple ways to build models (Sequential, Functional, Subclassing)
3. **Production-ready**: Easy deployment and serving
4. **Extensive ecosystem**: Large community and resources
5. **TensorFlow integration**: Seamless integration with TensorFlow ecosystem
6. **Pre-trained models**: Access to popular architectures
7. **Cross-platform**: Works on CPU, GPU, and TPU

## Limitations

1. **Abstraction overhead**: May hide important details
2. **Debugging complexity**: Can be harder to debug complex models
3. **Memory usage**: May use more memory than lower-level APIs
4. **Customization limits**: Some advanced features require TensorFlow knowledge
5. **Version compatibility**: API changes between versions

## Related Libraries

- **TensorFlow**: Backend and low-level operations
- **PyTorch**: Alternative deep learning framework
- **Scikit-learn**: Traditional machine learning
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive development
- **MLflow**: Experiment tracking
- **TensorBoard**: Training visualization 