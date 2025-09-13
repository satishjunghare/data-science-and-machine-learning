# Altair Library

## Overview
Altair is a declarative statistical visualization library for Python based on Vega-Lite. It provides a simple and intuitive API for creating beautiful, interactive visualizations with minimal code. Altair follows the grammar of graphics approach, making it easy to build complex charts by combining simple building blocks. It's particularly well-suited for exploratory data analysis and creating publication-quality visualizations.

## Installation
```bash
# Basic installation
pip install altair

# With additional dependencies
pip install "altair[all]"

# For saving charts
pip install altair_saver

# For Jupyter integration
pip install vega
```

## Key Features
- **Declarative Syntax**: Simple, readable chart specifications
- **Grammar of Graphics**: Consistent API based on data, marks, and encodings
- **Interactive Visualizations**: Built-in interactivity and selections
- **Vega-Lite Backend**: High-quality rendering and performance
- **Pandas Integration**: Seamless integration with pandas DataFrames
- **Faceted Plots**: Easy creation of multi-panel visualizations
- **Export Options**: Save charts in various formats
- **Jupyter Support**: Excellent integration with Jupyter notebooks

## Core Concepts

### Basic Chart Creation
```python
import altair as alt
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Basic scatter plot
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y'
)
chart

# Scatter plot with color encoding
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color='category'
)
chart
```

### Chart Components
```python
# Chart = Data + Mark + Encoding
chart = alt.Chart(df).mark_point().encode(
    x='x:Q',  # Quantitative data
    y='y:Q',
    color='category:N'  # Nominal data
)
chart
```

## Mark Types

### Point and Line Marks
```python
# Point marks
points = alt.Chart(df).mark_point().encode(
    x='x',
    y='y',
    color='category'
)

# Line marks
line = alt.Chart(df).mark_line().encode(
    x='x',
    y='y',
    color='category'
)

# Area marks
area = alt.Chart(df).mark_area().encode(
    x='x',
    y='y',
    color='category'
)
```

### Bar and Histogram Marks
```python
# Bar chart
bars = alt.Chart(df).mark_bar().encode(
    x='category',
    y='count()'
)

# Histogram
histogram = alt.Chart(df).mark_bar().encode(
    x=alt.X('x:Q', bin=True),
    y='count()'
)

# Grouped bar chart
grouped_bars = alt.Chart(df).mark_bar().encode(
    x='category',
    y='mean(y)',
    color='category'
)
```

### Text and Rule Marks
```python
# Text marks
text = alt.Chart(df).mark_text().encode(
    x='x',
    y='y',
    text='category'
)

# Rule marks (for reference lines)
rule = alt.Chart(df).mark_rule().encode(
    y='mean(y)',
    color='red'
)
```

## Data Encodings

### Position Encodings
```python
# X and Y encodings
chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('x:Q', title='X Axis', scale=alt.Scale(domain=[-3, 3])),
    y=alt.Y('y:Q', title='Y Axis', scale=alt.Scale(domain=[-3, 3]))
)
```

### Color and Size Encodings
```python
# Color encoding
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color=alt.Color('category:N', scale=alt.Scale(scheme='category10'))
)

# Size encoding
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    size=alt.Size('y:Q', scale=alt.Scale(range=[50, 300]))
)

# Combined encodings
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color='category',
    size='y',
    opacity=alt.value(0.7)
)
```

### Shape and Tooltip Encodings
```python
# Shape encoding
chart = alt.Chart(df).mark_point().encode(
    x='x',
    y='y',
    shape='category'
)

# Tooltip encoding
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    tooltip=['x', 'y', 'category']
)
```

## Statistical Transformations

### Aggregations
```python
# Mean aggregation
mean_chart = alt.Chart(df).mark_bar().encode(
    x='category',
    y='mean(y)'
)

# Multiple aggregations
agg_chart = alt.Chart(df).mark_bar().encode(
    x='category',
    y='mean(y)',
    color='category'
).transform_aggregate(
    mean_y='mean(y)',
    groupby=['category']
)
```

### Binning and Density
```python
# Binning
binned = alt.Chart(df).mark_bar().encode(
    x=alt.X('x:Q', bin=True),
    y='count()'
)

# Density estimation
density = alt.Chart(df).mark_area().encode(
    x=alt.X('x:Q', bin=True),
    y='density:Q'
).transform_density(
    density='x',
    as_=['x', 'density']
)
```

### Window Functions
```python
# Rolling mean
rolling = alt.Chart(df).mark_line().encode(
    x='x',
    y='rolling_mean:Q'
).transform_window(
    rolling_mean='mean(y)',
    frame=[-5, 5]
)
```

## Faceted Plots

### Faceted Charts
```python
# Faceted scatter plot
faceted = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color='category'
).facet(
    column='category'
)

# Faceted with different scales
faceted_scales = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y'
).facet(
    column='category',
    row='category'
).resolve_scale(
    x='independent',
    y='independent'
)
```

### Repeat and Layer
```python
# Repeat for multiple variables
repeated = alt.Chart(df).mark_circle().encode(
    x=alt.X(alt.repeat('column'), type='quantitative'),
    y=alt.Y(alt.repeat('row'), type='quantitative')
).repeat(
    column=['x', 'y'],
    row=['x', 'y']
)

# Layered charts
base = alt.Chart(df).encode(x='x', y='y')

layered = alt.layer(
    base.mark_circle(),
    base.mark_line()
)
```

## Interactive Features

### Selections
```python
# Single selection
selection = alt.selection_single()

chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color=alt.condition(selection, 'category', alt.value('lightgray'))
).add_selection(selection)

# Multi selection
multi_selection = alt.selection_multi()

chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color=alt.condition(multi_selection, 'category', alt.value('lightgray'))
).add_selection(multi_selection)
```

### Interval Selection
```python
# Interval selection
interval = alt.selection_interval()

chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color=alt.condition(interval, 'category', alt.value('lightgray'))
).add_selection(interval)
```

### Interactive Filters
```python
# Dropdown filter
dropdown = alt.binding_select(options=['A', 'B', 'C'])
selection = alt.selection_single(fields=['category'], bind=dropdown)

chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color='category'
).add_selection(selection).transform_filter(selection)
```

## Advanced Features

### Custom Scales and Axes
```python
# Custom color scale
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color=alt.Color('category', scale=alt.Scale(
        domain=['A', 'B', 'C'],
        range=['red', 'green', 'blue']
    ))
)

# Custom axis
chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('x', axis=alt.Axis(
        title='Custom X Title',
        titleColor='red',
        titleFontSize=14
    )),
    y='y'
)
```

### Conditional Encodings
```python
# Conditional color
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    color=alt.condition(
        alt.datum.y > 0,
        alt.value('red'),
        alt.value('blue')
    )
)

# Conditional size
chart = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y',
    size=alt.condition(
        alt.datum.x > 0,
        alt.value(100),
        alt.value(50)
    )
)
```

### Custom Transforms
```python
# Custom transform
chart = alt.Chart(df).mark_line().encode(
    x='x',
    y='y'
).transform_regression('x', 'y').mark_line(color='red')
```

## Export and Saving

### Save Charts
```python
# Save as PNG
chart.save('chart.png')

# Save as SVG
chart.save('chart.svg')

# Save as HTML
chart.save('chart.html')

# Save as JSON (Vega-Lite specification)
chart.save('chart.json')
```

### Display Options
```python
# Configure display
alt.renderers.enable('default')

# For Jupyter
alt.renderers.enable('jupyter')

# For HTML
alt.renderers.enable('html')
```

## Use Cases
- **Exploratory Data Analysis**: Quick visualization of data patterns
- **Statistical Visualization**: Create publication-quality statistical plots
- **Interactive Dashboards**: Build interactive data exploration tools
- **Research Presentations**: Create clear, informative visualizations
- **Data Storytelling**: Build narrative visualizations
- **Web Applications**: Embed interactive charts in web apps
- **Academic Papers**: Create precise, reproducible visualizations

## Best Practices
1. **Start Simple**: Begin with basic charts and add complexity gradually
2. **Use Appropriate Marks**: Choose marks that best represent your data
3. **Consider Color**: Use colorblind-friendly palettes and meaningful colors
4. **Add Interactivity**: Use selections and filters for exploration
5. **Facet When Needed**: Use faceting for comparing groups
6. **Optimize Performance**: Use appropriate data types and transformations
7. **Document Your Charts**: Add titles, labels, and descriptions

## Advantages
- **Declarative**: Simple, readable chart specifications
- **Consistent**: Grammar of graphics ensures consistency
- **Interactive**: Built-in interactivity and selections
- **Flexible**: Easy to customize and extend
- **Integration**: Works well with pandas and Jupyter
- **Export**: Multiple export formats available
- **Performance**: Efficient rendering with Vega-Lite

## Limitations
- **Learning Curve**: Understanding grammar of graphics concepts
- **Customization**: Limited compared to matplotlib for complex customizations
- **Performance**: May be slower for very large datasets
- **Dependencies**: Requires additional libraries for some features
- **Browser Dependency**: Interactive features require web browser

## Related Libraries
- **Vega-Lite**: Declarative visualization grammar
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Traditional plotting library
- **Seaborn**: Statistical plotting built on matplotlib
- **Plotly**: Interactive plotting library
- **Bokeh**: Interactive web plotting
- **Vega**: Visualization grammar and runtime 