# TensorFlow Library

## Overview
TensorFlow is an open-source machine learning framework developed by Google that enables developers to build and deploy machine learning models, particularly deep learning models. It provides a comprehensive ecosystem of tools, libraries, and community resources for numerical computation and large-scale machine learning. TensorFlow supports both CPU and GPU computation and is widely used in research and production environments.

## Installation
```bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA)
pip install tensorflow-gpu

# Latest version
pip install tensorflow==2.15.0
```

## Key Features
- **Deep Learning**: Comprehensive support for neural networks
- **GPU Acceleration**: Automatic GPU utilization for faster training
- **Keras Integration**: High-level API for easy model building
- **Production Ready**: Tools for model deployment and serving
- **TensorBoard**: Visualization and monitoring tools
- **Cross-Platform**: Support for multiple platforms and devices
- **AutoML**: Automated machine learning capabilities
- **Model Serving**: TensorFlow Serving for production deployment

## Core Concepts

### Tensors
```python
import tensorflow as tf
import numpy as np

# Create tensors
scalar = tf.constant(5)
vector = tf.constant([1, 2, 3, 4, 5])
matrix = tf.constant([[1, 2], [3, 4]])
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Scalar shape: {scalar.shape}")
print(f"Vector shape: {vector.shape}")
print(f"Matrix shape: {matrix.shape}")
print(f"3D Tensor shape: {tensor_3d.shape}")

# Tensor operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b  # Element-wise addition
d = tf.matmul(tf.reshape(a, [3, 1]), tf.reshape(b, [1, 3]))  # Matrix multiplication
```

### Variables and Operations
```python
# Variables (mutable tensors)
var = tf.Variable([1.0, 2.0, 3.0])
print(f"Variable: {var}")

# Update variable
var.assign([4.0, 5.0, 6.0])
print(f"Updated variable: {var}")

# Mathematical operations
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1

# Compute gradient
dy_dx = tape.gradient(y, x)
print(f"dy/dx = {dy_dx}")
```

## Building Neural Networks

### Sequential Model (Keras API)
```python
from tensorflow import keras
from tensorflow.keras import layers

# Create sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()
```

### Functional API
```python
# Input layer
inputs = keras.Input(shape=(784,))

# Hidden layers
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Output layer
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Custom Layers
```python
class CustomDense(layers.Layer):
    def __init__(self, units=32, activation=None):
        super(CustomDense, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# Use custom layer
model = keras.Sequential([
    CustomDense(128, activation='relu', input_shape=(784,)),
    CustomDense(10, activation='softmax')
])
```

## Training Models

### Basic Training
```python
# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Train model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Custom Training Loop
```python
# Define optimizer and loss
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}/10")
    
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)
        
        # Backward pass
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss_value:.4f}")
```

### Callbacks
```python
# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train with callbacks
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks
)
```

## Convolutional Neural Networks (CNNs)

### CNN for Image Classification
```python
# Build CNN
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

# Prepare data for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_cnn, y_train, epochs=5, validation_split=0.2)
```

## Recurrent Neural Networks (RNNs)

### LSTM for Sequence Data
```python
# Build LSTM model
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(None, 1)),
    layers.LSTM(64),
    layers.Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Generate sequence data
def generate_sequence_data(n_samples, seq_length):
    x = np.random.randn(n_samples, seq_length, 1)
    y = np.sum(x, axis=1)  # Sum of sequence
    return x, y

x_seq, y_seq = generate_sequence_data(1000, 10)

# Train
model.fit(x_seq, y_seq, epochs=10, validation_split=0.2)
```

## Transfer Learning

### Using Pre-trained Models
```python
# Load pre-trained model
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Model Deployment

### Save and Load Models
```python
# Save model
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save model in SavedModel format
model.save('my_model_savedmodel')

# Load SavedModel
loaded_model = tf.keras.models.load_model('my_model_savedmodel')
```

### TensorFlow Serving
```python
# Convert model to TensorFlow Serving format
import tensorflow as tf

# Save model for serving
model.save('serving_model/1', save_format='tf')

# The model can now be served using TensorFlow Serving
# docker run -p 8501:8501 --mount type=bind,source=/path/to/serving_model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving
```

### Model Optimization
```python
# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save optimized model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## TensorBoard Visualization

### Logging Training Metrics
```python
# Create TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Train with TensorBoard
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[tensorboard_callback]
)

# Launch TensorBoard
# tensorboard --logdir=./logs
```

### Custom Metrics Logging
```python
# Custom callback for logging
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log custom metrics
        tf.summary.scalar('custom_metric', logs['accuracy'], step=epoch)

# Use custom callback
model.fit(
    x_train, y_train,
    epochs=10,
    callbacks=[CustomCallback()]
)
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
6. **Callbacks**: Use callbacks for early stopping and model checkpointing
7. **GPU Utilization**: Monitor GPU usage and optimize batch sizes
8. **Model Saving**: Save models regularly and use version control

## Advantages
- **Comprehensive**: Full ecosystem for deep learning
- **Production Ready**: Tools for deployment and serving
- **GPU Support**: Excellent GPU acceleration
- **Community**: Large, active community and extensive documentation
- **Flexibility**: Both high-level and low-level APIs
- **Integration**: Works well with other Google tools
- **Scalability**: Can handle large-scale distributed training

## Limitations
- **Learning Curve**: Steep learning curve for beginners
- **Memory Usage**: Can be memory-intensive for large models
- **Debugging**: Complex debugging for custom operations
- **Version Compatibility**: Frequent API changes between versions
- **Mobile Deployment**: Limited compared to specialized frameworks

## Related Libraries
- **Keras**: High-level API for TensorFlow
- **TensorFlow Hub**: Pre-trained models and modules
- **TensorFlow Extended (TFX)**: Production ML pipeline platform
- **TensorFlow Lite**: Mobile and edge deployment
- **TensorFlow Serving**: Model serving infrastructure
- **PyTorch**: Alternative deep learning framework
- **NumPy**: Numerical computing foundation 