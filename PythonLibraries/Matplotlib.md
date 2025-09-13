# Matplotlib Library

## Overview
Matplotlib is the foundation of plotting in Python and is highly customizable. It serves as the base plotting library upon which other visualization libraries like Seaborn, Plotly, and Bokeh are built. Matplotlib provides complete control over every aspect of a plot, making it the go-to library for creating publication-quality graphics.

## Installation
```bash
pip install matplotlib
```

## Key Features
- **Complete Control**: Fine-grained control over every plot element
- **Multiple Backends**: Support for various output formats (GUI, file, web)
- **Publication Quality**: High-resolution output suitable for academic papers
- **Extensive Customization**: Control over colors, fonts, layouts, and styling
- **Integration**: Works seamlessly with NumPy, Pandas, and other scientific libraries
- **Cross-Platform**: Runs on Windows, macOS, and Linux

## Core Functionality
- **2D Plotting**: Line plots, scatter plots, bar charts, histograms
- **3D Plotting**: Surface plots, wireframes, 3D scatter plots
- **Statistical Plots**: Box plots, violin plots, error bars
- **Image Processing**: Display and manipulate images
- **Animation**: Create animated plots and visualizations
- **Subplots**: Create complex multi-panel figures

## Common Plot Types

### Basic 2D Plots
```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y, alpha=0.6, c='red', s=50)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot')
plt.show()
```

### Bar and Histogram Plots
```python
# Bar chart
categories = ['A', 'B', 'C', 'D']
values = [4, 3, 2, 1]
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

### Subplots and Complex Layouts
```python
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Subplot 1: Line plot
x = np.linspace(0, 2*np.pi, 100)
axes[0,0].plot(x, np.sin(x), 'b-', label='sin(x)')
axes[0,0].plot(x, np.cos(x), 'r-', label='cos(x)')
axes[0,0].set_title('Trigonometric Functions')
axes[0,0].legend()
axes[0,0].grid(True)

# Subplot 2: Scatter plot
x = np.random.randn(50)
y = np.random.randn(50)
axes[0,1].scatter(x, y, alpha=0.6)
axes[0,1].set_title('Scatter Plot')

# Subplot 3: Bar plot
categories = ['A', 'B', 'C']
values = [3, 7, 5]
axes[1,0].bar(categories, values, color=['red', 'blue', 'green'])
axes[1,0].set_title('Bar Plot')

# Subplot 4: Histogram
data = np.random.randn(1000)
axes[1,1].hist(data, bins=20, alpha=0.7)
axes[1,1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### Advanced Customization
```python
# Custom styling
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(10, 6))

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot with custom styling
ax.plot(x, y1, 'o-', color='#2E86AB', linewidth=2, markersize=4, 
        label='Sine', alpha=0.8)
ax.plot(x, y2, 's-', color='#A23B72', linewidth=2, markersize=4, 
        label='Cosine', alpha=0.8)

# Customize appearance
ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
ax.set_title('Wave Functions', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

# Add text annotation
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.show()
```

## Use Cases
- **Scientific Research**: Publication-quality figures for academic papers
- **Data Analysis**: Exploratory data visualization and analysis
- **Business Intelligence**: Creating dashboards and reports
- **Engineering**: Technical drawings and simulations
- **Education**: Teaching and learning materials
- **Web Development**: Generating plots for web applications

## Best Practices
1. **Figure Size**: Set appropriate figure sizes for your output medium
2. **Color Schemes**: Use colorblind-friendly palettes when possible
3. **Typography**: Choose readable fonts and appropriate sizes
4. **Layout**: Use `plt.tight_layout()` to avoid overlapping elements
5. **Resolution**: Set high DPI for publication-quality output
6. **Consistency**: Maintain consistent styling across related plots

## Saving and Exporting
```python
# Save in different formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plot.svg', dpi=300, bbox_inches='tight')

# High-resolution for publications
plt.savefig('publication_plot.png', dpi=600, bbox_inches='tight')
```

## Integration with Other Libraries
```python
# With Pandas
import pandas as pd
df = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})
df.plot.scatter(x='x', y='y')

# With NumPy
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)

# With Seaborn (built on matplotlib)
import seaborn as sns
sns.set_style("whitegrid")
plt.plot(x, y)
```

## Advantages
- **Complete Control**: Fine-grained control over every plot element
- **Flexibility**: Can create virtually any type of visualization
- **Maturity**: Well-established library with extensive documentation
- **Integration**: Works with all major scientific Python libraries
- **Publication Ready**: High-quality output suitable for academic papers
- **Cross-Platform**: Consistent behavior across different operating systems

## Limitations
- **Learning Curve**: Steep learning curve for complex customizations
- **Verbose**: Requires more code for simple plots compared to high-level libraries
- **Default Styling**: Basic default appearance requires customization
- **Performance**: Can be slow for very large datasets
- **Complexity**: Overwhelming for beginners who just need simple plots

## Related Libraries
- **Seaborn**: High-level statistical plotting built on matplotlib
- **Plotly**: Interactive web-based visualizations
- **Bokeh**: Interactive web visualizations with JavaScript backend
- **Altair**: Declarative statistical visualization
- **Pandas**: Built-in plotting methods using matplotlib backend