# OpenCV Library

## Overview
OpenCV (Open Source Computer Vision Library) is a comprehensive computer vision and machine learning software library. It provides a wide range of algorithms for image and video processing, including image filtering, feature detection, object recognition, camera calibration, and machine learning utilities. OpenCV is widely used in robotics, augmented reality, medical imaging, security systems, and many other applications.

## Installation
```bash
# Basic installation
pip install opencv-python

# Full installation with extra modules
pip install opencv-python-headless

# With contrib modules
pip install opencv-contrib-python

# Latest version
pip install opencv-python==4.8.1.78

# From conda
conda install -c conda-forge opencv
```

## Key Features
- **Image Processing**: Comprehensive image manipulation and filtering
- **Video Processing**: Video capture, playback, and analysis
- **Feature Detection**: Corner, edge, and blob detection
- **Object Recognition**: Face detection, object tracking
- **Machine Learning**: Built-in ML algorithms and utilities
- **Camera Calibration**: Camera intrinsic and extrinsic calibration
- **GUI Support**: High-level GUI for image display and interaction
- **Cross-platform**: Works on Windows, macOS, and Linux

## Core Concepts

### Basic Image Operations
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('image.jpg')
print(f"Image shape: {image.shape}")

# Display image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to different color spaces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save image
cv2.imwrite('output.jpg', gray)

# Create blank image
blank = np.zeros((300, 300, 3), dtype=np.uint8)
blank[:] = (255, 0, 0)  # Blue background
```

### Image Manipulation
```python
# Resize image
resized = cv2.resize(image, (640, 480))
resized_proportional = cv2.resize(image, None, fx=0.5, fy=0.5)

# Rotate image
height, width = image.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

# Crop image
cropped = image[100:300, 200:400]  # y:y+h, x:x+w

# Flip image
flipped_horizontal = cv2.flip(image, 1)  # 1 for horizontal
flipped_vertical = cv2.flip(image, 0)    # 0 for vertical
flipped_both = cv2.flip(image, -1)       # -1 for both
```

## Image Filtering

### Basic Filters
```python
# Blur filters
blur = cv2.blur(image, (5, 5))
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
median_blur = cv2.medianBlur(image, 5)
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Sharpening kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, kernel)

# Edge detection
edges = cv2.Canny(image, 100, 200)
```

### Morphological Operations
```python
# Create kernel
kernel = np.ones((5, 5), np.uint8)

# Erosion and dilation
erosion = cv2.erode(image, kernel, iterations=1)
dilation = cv2.dilate(image, kernel, iterations=1)

# Opening and closing
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Gradient
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
```

## Feature Detection

### Corner Detection
```python
# Harris corner detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate to mark corners
corners = cv2.dilate(corners, None)

# Threshold
image[corners > 0.01 * corners.max()] = [0, 0, 255]

# Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)
```

### Edge Detection
```python
# Sobel edge detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.sqrt(sobelx**2 + sobely**2)

# Laplacian edge detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
```

## Object Detection

### Face Detection
```python
# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray)

for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

### Template Matching
```python
# Load template and image
template = cv2.imread('template.jpg', 0)
image = cv2.imread('image.jpg', 0)

# Template matching
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Find best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

# Draw rectangle
cv2.rectangle(image, top_left, bottom_right, 255, 2)
```

## Video Processing

### Video Capture
```python
# Open camera
cap = cv2.VideoCapture(0)  # 0 for default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Read video from file
cap = cv2.VideoCapture('video.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Frame count: {frame_count}")
print(f"Resolution: {width}x{height}")
```

### Video Processing Loop
```python
# Video processing
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video")
        break
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply some processing
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Display frame
    cv2.imshow('Processed Video', blurred)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Video Writing
```python
# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Write frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    
    # Write frame
    out.write(processed_frame)
    
    # Display
    cv2.imshow('Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## Drawing and Text

### Drawing Functions
```python
# Create blank image
canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# Draw lines
cv2.line(canvas, (0, 0), (600, 400), (255, 0, 0), 5)

# Draw rectangles
cv2.rectangle(canvas, (50, 50), (200, 150), (0, 255, 0), 3)
cv2.rectangle(canvas, (250, 50), (400, 150), (0, 255, 0), -1)  # Filled

# Draw circles
cv2.circle(canvas, (300, 300), 50, (0, 0, 255), 3)
cv2.circle(canvas, (450, 300), 50, (0, 0, 255), -1)  # Filled

# Draw ellipses
cv2.ellipse(canvas, (100, 300), (50, 30), 0, 0, 180, (255, 255, 0), 3)

# Draw polygons
pts = np.array([[350, 200], [400, 250], [350, 300], [300, 250]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(canvas, [pts], True, (255, 0, 255), 3)
```

### Text and Fonts
```python
# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(canvas, 'OpenCV Text', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Get text size
text_size = cv2.getTextSize('OpenCV Text', font, 1, 2)
print(f"Text size: {text_size}")

# Draw text with background
text = "Hello OpenCV"
text_size = cv2.getTextSize(text, font, 1, 2)[0]
text_x = 10
text_y = 100

# Draw background rectangle
cv2.rectangle(canvas, (text_x, text_y - text_size[1]), 
              (text_x + text_size[0], text_y + 10), (0, 0, 0), -1)

# Draw text
cv2.putText(canvas, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
```

## Machine Learning Integration

### Using Pre-trained Models
```python
# Load pre-trained model (example with DNN)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

# Prepare input
blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123), False, False)

# Forward pass
net.setInput(blob)
output = net.forward()

# Process output
print(f"Output shape: {output.shape}")
```

### K-means Clustering
```python
# Reshape image for clustering
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Apply k-means
k = 3
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to uint8
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)
```

## Camera Calibration

### Camera Calibration
```python
# Prepare object points
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

if ret:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    
    # Draw corners
    cv2.drawChessboardCorners(image, (7, 6), corners2, ret)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

## Use Cases
- **Image Processing**: Filtering, enhancement, and transformation
- **Computer Vision**: Object detection, tracking, and recognition
- **Video Analysis**: Motion detection, surveillance, and analysis
- **Medical Imaging**: Image processing for medical applications
- **Robotics**: Vision systems for autonomous robots
- **Augmented Reality**: Real-time image processing and overlay
- **Security Systems**: Face recognition and surveillance
- **Industrial Inspection**: Quality control and defect detection

## Best Practices
1. **Memory Management**: Release resources properly
2. **Error Handling**: Check return values and handle errors
3. **Performance**: Use appropriate data types and algorithms
4. **Image Formats**: Choose appropriate formats for your use case
5. **Camera Settings**: Configure camera parameters for optimal results
6. **Multi-threading**: Use threading for real-time applications
7. **GPU Acceleration**: Use GPU when available for better performance
8. **Code Optimization**: Profile and optimize critical sections

## Advantages
- **Comprehensive**: Wide range of computer vision algorithms
- **Performance**: Optimized C++ backend with Python bindings
- **Cross-platform**: Works on multiple operating systems
- **Active Community**: Large community and regular updates
- **Documentation**: Extensive documentation and examples
- **Integration**: Easy integration with other libraries
- **Real-time**: Suitable for real-time applications

## Limitations
- **Learning Curve**: Complex API with many functions
- **Memory Usage**: Can be memory-intensive for large images/videos
- **Performance**: Python bindings may be slower than C++
- **Dependencies**: Requires additional libraries for some features
- **Platform Specific**: Some features may work differently across platforms

## Related Libraries
- **NumPy**: Numerical computing foundation
- **Matplotlib**: Plotting and visualization
- **PIL/Pillow**: Image processing
- **Scikit-image**: Scientific image processing
- **TensorFlow**: Deep learning for computer vision
- **PyTorch**: Alternative deep learning framework
- **MediaPipe**: Media processing and ML pipelines 