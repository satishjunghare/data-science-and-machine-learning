# Bokeh Library

## Overview
Bokeh is a powerful Python library for creating interactive web visualizations and dashboards. It provides both high-level and low-level APIs for building sophisticated plots, charts, and applications that can be embedded in web browsers. Bokeh excels at creating interactive data applications with real-time updates, making it ideal for building data dashboards and web-based analytics tools.

## Installation
```bash
# Basic installation
pip install bokeh

# With additional dependencies
pip install "bokeh[all]"

# For Jupyter integration
pip install jupyter_bokeh

# For server applications
pip install bokeh-server
```

## Key Features
- **Interactive Visualizations**: Rich interactivity with zoom, pan, hover, and selection tools
- **Web-Based**: Built for web deployment and browser rendering
- **Real-time Updates**: Support for streaming and live data updates
- **Multiple Output Formats**: HTML, PNG, SVG, and PDF export
- **Server Applications**: Create interactive web applications with Bokeh server
- **High-Level and Low-Level APIs**: Both simple and complex plotting capabilities
- **JavaScript Integration**: Custom JavaScript callbacks and extensions
- **Responsive Design**: Adapts to different screen sizes and devices

## Core Components

### Basic Plotting
```python
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import numpy as np

# Enable notebook output
output_notebook()

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure
p = figure(width=600, height=400, title='Sine Wave')

# Add line glyph
p.line(x, y, line_width=2, line_color='blue')

# Add scatter points
p.circle(x[::10], y[::10], size=8, color='red', alpha=0.6)

# Show plot
show(p)
```

### Multiple Glyphs
```python
from bokeh.plotting import figure, show
import numpy as np

# Create data
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.choice(['red', 'blue', 'green'], 100)
sizes = np.random.uniform(5, 20, 100)

# Create figure
p = figure(width=600, height=400, title='Scatter Plot')

# Add multiple glyphs
p.circle(x, y, size=sizes, color=colors, alpha=0.6)
p.square(x[::2], y[::2], size=10, color='black', alpha=0.8)

# Customize appearance
p.xaxis.axis_label = 'X Values'
p.yaxis.axis_label = 'Y Values'
p.grid.grid_line_alpha = 0.3

show(p)
```

## Interactive Features

### Hover Tools
```python
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
import numpy as np

# Create data
x = np.random.randn(100)
y = np.random.randn(100)
categories = np.random.choice(['A', 'B', 'C'], 100)

# Create figure
p = figure(width=600, height=400, title='Interactive Scatter Plot')

# Add hover tool
hover = HoverTool(tooltips=[
    ('x', '@x'),
    ('y', '@y'),
    ('category', '@category')
])

p.add_tools(hover)

# Add scatter plot with data source
from bokeh.models import ColumnDataSource
source = ColumnDataSource(data=dict(x=x, y=y, category=categories))

p.circle('x', 'y', source=source, size=10, alpha=0.6)

show(p)
```

### Selection Tools
```python
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import numpy as np

# Create data
x = np.random.randn(100)
y = np.random.randn(100)
colors = ['red'] * 100

source = ColumnDataSource(data=dict(x=x, y=y, color=colors))

# Create figure with selection tools
p = figure(width=600, height=400, 
           tools='box_select,lasso_select,pan,wheel_zoom,reset')

# Add scatter plot
p.circle('x', 'y', source=source, size=10, color='color', alpha=0.6)

# Add callback for selection
from bokeh.models import CustomJS

callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var selected = source.selected.indices;
    
    // Change color of selected points
    for (var i = 0; i < data.color.length; i++) {
        data.color[i] = 'red';
    }
    for (var i = 0; i < selected.length; i++) {
        data.color[selected[i]] = 'blue';
    }
    
    source.change.emit();
""")

source.selected.js_on_change('indices', callback)

show(p)
```

## Layouts and Widgets

### Layouts
```python
from bokeh.layouts import row, column, gridplot
from bokeh.plotting import figure, show
import numpy as np

# Create multiple plots
p1 = figure(width=300, height=300, title='Plot 1')
p2 = figure(width=300, height=300, title='Plot 2')
p3 = figure(width=300, height=300, title='Plot 3')
p4 = figure(width=300, height=300, title='Plot 4')

# Add data to plots
x = np.linspace(0, 10, 100)
p1.line(x, np.sin(x))
p2.circle(x, np.cos(x))
p3.square(x, np.tan(x))
p4.triangle(x, np.exp(-x))

# Create layouts
row_layout = row(p1, p2)
column_layout = column(p1, p2)
grid_layout = gridplot([[p1, p2], [p3, p4]])

show(grid_layout)
```

### Widgets
```python
from bokeh.layouts import column
from bokeh.models import Slider, Button, Select
from bokeh.plotting import figure, show
import numpy as np

# Create plot
p = figure(width=600, height=400, title='Interactive Plot')
x = np.linspace(0, 10, 100)
line = p.line(x, np.sin(x), line_width=2)

# Create widgets
slider = Slider(start=0, end=10, value=1, step=0.1, title='Frequency')
button = Button(label='Reset', button_type='success')
select = Select(title='Function', options=['sin', 'cos', 'tan'], value='sin')

# Add callback
from bokeh.models import CustomJS

callback = CustomJS(args=dict(line=line, slider=slider, select=select), code="""
    var freq = slider.value;
    var func = select.value;
    var x = [];
    var y = [];
    
    for (var i = 0; i <= 100; i++) {
        x.push(i * 0.1);
        if (func === 'sin') {
            y.push(Math.sin(i * 0.1 * freq));
        } else if (func === 'cos') {
            y.push(Math.cos(i * 0.1 * freq));
        } else {
            y.push(Math.tan(i * 0.1 * freq));
        }
    }
    
    line.data_source.data = {x: x, y: y};
    line.data_source.change.emit();
""")

slider.js_on_change('value', callback)
select.js_on_change('value', callback)

# Create layout
layout = column(slider, select, button, p)
show(layout)
```

## Data Sources

### ColumnDataSource
```python
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import pandas as pd
import numpy as np

# Create DataFrame
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.uniform(5, 20, 100)
})

# Convert to ColumnDataSource
source = ColumnDataSource(df)

# Create plot
p = figure(width=600, height=400, title='DataFrame Plot')

# Add glyphs using column names
p.circle('x', 'y', source=source, size='size', 
         color='category', alpha=0.6)

show(p)
```

### CDSView for Filtering
```python
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CDSView, IndexFilter
import numpy as np

# Create data
x = np.random.randn(100)
y = np.random.randn(100)
categories = np.random.choice(['A', 'B', 'C'], 100)

source = ColumnDataSource(data=dict(x=x, y=y, category=categories))

# Create view for category A
view_a = CDSView(source=source, 
                 filters=[IndexFilter([i for i, c in enumerate(categories) if c == 'A'])])

# Create plot
p = figure(width=600, height=400, title='Filtered Plot')

# Add glyphs with different views
p.circle('x', 'y', source=source, view=view_a, 
         size=10, color='red', alpha=0.6, legend_label='Category A')

show(p)
```

## Advanced Plotting

### Statistical Plots
```python
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import numpy as np

# Create data
x = np.random.randn(1000)
y = np.random.randn(1000)

# Create hexbin plot
p = figure(width=600, height=400, title='Hexbin Plot')
p.hexbin(x, y, size=0.1, palette='Viridis256')

show(p)

# Create histogram
from bokeh.layouts import row

hist, edges = np.histogram(x, bins=30)
p2 = figure(width=600, height=400, title='Histogram')
p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], 
        fill_color='steelblue', alpha=0.6)

show(row(p, p2))
```

### Geographic Plots
```python
from bokeh.plotting import figure, show
from bokeh.tile_providers import CARTODBPOSITRON

# Create map
p = figure(width=600, height=400, 
           x_range=(-2000000, 2000000), 
           y_range=(-1000000, 1000000),
           title='Map Plot')

# Add tile source
p.add_tile(CARTODBPOSITRON)

# Add points (example coordinates)
x = [-1000000, 0, 1000000]
y = [0, 0, 0]
p.circle(x, y, size=20, color='red', alpha=0.6)

show(p)
```

## Bokeh Server Applications

### Basic Server App
```python
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
source = ColumnDataSource(data=dict(x=x, y=np.sin(x)))

# Create plot
p = figure(width=600, height=400, title='Server Plot')
p.line('x', 'y', source=source, line_width=2)

# Create slider
slider = Slider(start=0, end=10, value=1, step=0.1, title='Frequency')

# Define callback
def update(attr, old, new):
    freq = slider.value
    source.data = dict(x=x, y=np.sin(freq * x))

slider.on_change('value', update)

# Add to document
curdoc().add_root(column(slider, p))
curdoc().title = "Bokeh Server App"
```

### Real-time Updates
```python
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import numpy as np
from datetime import datetime, timedelta

# Create data
dates = [datetime.now() + timedelta(minutes=i) for i in range(100)]
values = np.random.randn(100).cumsum()

source = ColumnDataSource(data=dict(x=dates, y=values))

# Create plot
p = figure(width=800, height=400, x_axis_type='datetime', title='Real-time Data')
p.line('x', 'y', source=source, line_width=2)

# Update function
def update():
    new_date = datetime.now()
    new_value = values[-1] + np.random.randn()
    
    dates.append(new_date)
    values.append(new_value)
    
    # Keep only last 100 points
    if len(dates) > 100:
        dates.pop(0)
        values.pop(0)
    
    source.data = dict(x=dates, y=values)

# Add periodic callback
curdoc().add_periodic_callback(update, 1000)  # Update every second
curdoc().add_root(column(p))
```

## Export and Embedding

### Export Options
```python
from bokeh.plotting import figure, show
from bokeh.io import export_png, export_svg
import numpy as np

# Create plot
p = figure(width=600, height=400, title='Export Example')
x = np.linspace(0, 10, 100)
p.line(x, np.sin(x), line_width=2)

# Export to different formats
export_png(p, filename='plot.png')
export_svg(p, filename='plot.svg')

show(p)
```

### Embedding in Web Pages
```python
from bokeh.plotting import figure
from bokeh.embed import components, file_html
from bokeh.resources import CDN
import numpy as np

# Create plot
p = figure(width=600, height=400, title='Embedded Plot')
x = np.linspace(0, 10, 100)
p.line(x, np.sin(x), line_width=2)

# Generate components for web embedding
script, div = components(p)

# Generate complete HTML
html = file_html(p, CDN, "My Plot")

print("Script:", script)
print("Div:", div)
```

## Use Cases
- **Interactive Dashboards**: Create dynamic data dashboards
- **Real-time Monitoring**: Live data visualization and monitoring
- **Web Applications**: Embed interactive plots in web apps
- **Data Exploration**: Interactive data analysis tools
- **Scientific Visualization**: Complex scientific plots and charts
- **Business Intelligence**: Interactive business reports
- **Educational Tools**: Interactive learning visualizations

## Best Practices
1. **Use ColumnDataSource**: For efficient data handling and updates
2. **Optimize Performance**: Use appropriate data structures and update methods
3. **Responsive Design**: Make layouts adapt to different screen sizes
4. **Interactive Elements**: Add meaningful interactivity for better UX
5. **Error Handling**: Implement proper error handling for data updates
6. **Documentation**: Document complex callbacks and interactions
7. **Testing**: Test interactive features thoroughly

## Advantages
- **Rich Interactivity**: Advanced interactive features and tools
- **Web-Native**: Built for web deployment and browser rendering
- **Real-time Updates**: Excellent support for live data updates
- **Flexible Layouts**: Powerful layout system for complex applications
- **JavaScript Integration**: Custom JavaScript callbacks and extensions
- **Server Applications**: Full-featured server for interactive apps
- **Multiple Outputs**: Support for various export formats

## Limitations
- **Learning Curve**: Complex API for advanced features
- **Performance**: May be slower for very large datasets
- **Dependencies**: Requires additional libraries for some features
- **Browser Compatibility**: Some features may not work in older browsers
- **Setup Complexity**: Server applications require additional setup

## Related Libraries
- **Plotly**: Alternative interactive plotting library
- **Dash**: Web application framework for data science
- **Streamlit**: Rapid web app development
- **Matplotlib**: Traditional plotting library
- **Seaborn**: Statistical plotting built on matplotlib
- **Altair**: Declarative visualization library
- **Vega-Lite**: Declarative visualization grammar 