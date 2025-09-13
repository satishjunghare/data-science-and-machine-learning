# Plotly Library

## Overview
Plotly is a comprehensive library for creating interactive, publication-quality graphs and dashboards in Python. It provides both high-level and low-level APIs for creating a wide variety of charts, from simple line plots to complex 3D visualizations. Plotly excels at creating web-based interactive visualizations that can be easily shared and embedded in web applications.

## Installation
```bash
pip install plotly
```

## Key Features
- **Interactive Visualizations**: Hover effects, zoom, pan, and selection tools
- **Multiple Output Formats**: HTML, PNG, SVG, and PDF export
- **Web Integration**: Built for web deployment and sharing
- **Rich Chart Types**: Line, scatter, bar, histogram, 3D, and more
- **Dashboards**: Create interactive dashboards with Dash
- **Real-time Updates**: Support for live data streaming
- **Customization**: Extensive styling and layout options
- **Export Options**: Save as static images or interactive HTML

## Core Components

### Basic Plotting with Plotly Express
```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.uniform(10, 100, 100)
})

# Scatter plot
fig = px.scatter(df, x='x', y='y', color='category', size='size',
                 title='Interactive Scatter Plot')
fig.show()

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = px.line(x=x, y=y, title='Sine Wave')
fig.show()
```

### Advanced Plotting with Graph Objects
```python
# Create figure with multiple traces
fig = go.Figure()

# Add scatter trace
fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    name='Scatter',
    marker=dict(
        size=10,
        color=df['category'],
        colorscale='Viridis',
        showscale=True
    )
))

# Add line trace
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='lines',
    name='Line',
    line=dict(color='red', width=2)
))

# Update layout
fig.update_layout(
    title='Multiple Traces',
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    template='plotly_white'
)

fig.show()
```

## Chart Types

### Statistical Charts
```python
# Histogram
fig = px.histogram(df, x='x', nbins=30, title='Distribution of X')
fig.show()

# Box plot
fig = px.box(df, x='category', y='y', title='Box Plot by Category')
fig.show()

# Violin plot
fig = px.violin(df, x='category', y='y', title='Violin Plot')
fig.show()

# Density heatmap
fig = px.density_heatmap(df, x='x', y='y', title='Density Heatmap')
fig.show()
```

### 3D Visualizations
```python
# 3D scatter plot
fig = px.scatter_3d(df, x='x', y='y', z='size', color='category',
                    title='3D Scatter Plot')
fig.show()

# 3D surface plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
fig.update_layout(title='3D Surface Plot', scene_aspectmode='cube')
fig.show()
```

### Geographic Plots
```python
# Choropleth map
fig = px.choropleth(
    locations=['USA', 'Canada', 'Mexico'],
    locationmode='country names',
    color=[1, 2, 3],
    title='Sample Choropleth Map'
)
fig.show()

# Scatter mapbox
fig = px.scatter_mapbox(
    df,
    lat=df['x'],  # Using x as latitude for demo
    lon=df['y'],  # Using y as longitude for demo
    color='category',
    title='Scatter Map'
)
fig.update_layout(mapbox_style='open-street-map')
fig.show()
```

## Interactive Features

### Hover Information
```python
# Custom hover template
fig = px.scatter(df, x='x', y='y', color='category',
                 hover_data=['size'],
                 hover_name='category')

fig.update_traces(
    hovertemplate="<b>%{hover_name}</b><br>" +
                  "X: %{x:.2f}<br>" +
                  "Y: %{y:.2f}<br>" +
                  "Size: %{customdata[0]:.1f}<br>" +
                  "<extra></extra>"
)

fig.show()
```

### Selection and Filtering
```python
# Scatter plot with selection
fig = px.scatter(df, x='x', y='y', color='category')

# Add selection
fig.update_layout(
    dragmode='select',
    selectdirection='any'
)

fig.show()
```

### Subplots and Facets
```python
# Faceted scatter plot
fig = px.scatter(df, x='x', y='y', color='category',
                 facet_col='category', facet_col_wrap=2,
                 title='Faceted Scatter Plot')
fig.show()

# Subplots with different chart types
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Scatter', 'Histogram', 'Box Plot', 'Line'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Add traces
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['x']), row=1, col=2)
fig.add_trace(go.Box(y=df['y']), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=y), row=2, col=2)

fig.update_layout(height=600, title_text="Multiple Subplots")
fig.show()
```

## Styling and Customization

### Themes and Templates
```python
# Available templates
import plotly.io as pio
print(pio.templates)

# Use different template
fig = px.scatter(df, x='x', y='y')
fig.update_layout(template='plotly_dark')
fig.show()

# Custom theme
fig.update_layout(
    template='plotly_white',
    font=dict(family="Arial", size=12, color="black"),
    plot_bgcolor='white',
    paper_bgcolor='white'
)
```

### Color Scales and Palettes
```python
# Custom color scale
fig = px.scatter(df, x='x', y='y', color='size',
                 color_continuous_scale='Viridis')

# Discrete color sequence
fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=['red', 'blue', 'green'])

fig.show()
```

### Layout Customization
```python
fig = px.scatter(df, x='x', y='y')

fig.update_layout(
    title={
        'text': 'Custom Title',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='Custom X Label',
    yaxis_title='Custom Y Label',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    width=800,
    height=600
)

fig.show()
```

## Export and Sharing

### Static Image Export
```python
# Export as PNG
fig.write_image("plot.png", width=800, height=600)

# Export as SVG
fig.write_image("plot.svg")

# Export as PDF
fig.write_image("plot.pdf")
```

### HTML Export
```python
# Save as interactive HTML
fig.write_html("interactive_plot.html")

# Include in Jupyter notebook
fig.show()

# Embed in web page
html_string = fig.to_html(include_plotlyjs=True)
```

## Dash Integration

### Creating Dashboards
```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1('Interactive Dashboard'),
    
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': cat, 'value': cat} for cat in df['category'].unique()],
        value=df['category'].unique()[0]
    ),
    
    dcc.Graph(id='scatter-plot')
])

# Callback to update plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('category-dropdown', 'value')
)
def update_graph(selected_category):
    filtered_df = df[df['category'] == selected_category]
    fig = px.scatter(filtered_df, x='x', y='y', title=f'Data for {selected_category}')
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
```

## Advanced Features

### Animation
```python
# Animated scatter plot
fig = px.scatter(df, x='x', y='y', color='category',
                 animation_frame='category',
                 title='Animated Scatter Plot')
fig.show()

# Animated line plot
fig = px.line(df, x='x', y='y', animation_frame='category')
fig.show()
```

### Statistical Annotations
```python
# Add trend line
fig = px.scatter(df, x='x', y='y', trendline='ols')
fig.show()

# Add confidence intervals
fig = px.scatter(df, x='x', y='y', trendline='ols',
                 trendline_options=dict(log_x=True))
fig.show()
```

### Custom JavaScript
```python
# Add custom JavaScript for interactivity
fig = px.scatter(df, x='x', y='y')

fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True, False]}],
                    label="Show Points",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Show Line",
                    method="update"
                )
            ]),
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ]
)

fig.show()
```

## Use Cases
- **Data Exploration**: Interactive exploration of datasets
- **Web Dashboards**: Create interactive web applications
- **Scientific Visualization**: Publication-quality scientific plots
- **Business Intelligence**: Interactive business reports
- **Real-time Monitoring**: Live data visualization
- **Educational Content**: Interactive learning materials
- **Research Presentations**: Dynamic research presentations

## Best Practices
1. **Choose Appropriate Chart Types**: Select charts that best represent your data
2. **Use Interactive Features**: Leverage hover, zoom, and selection tools
3. **Consistent Styling**: Maintain consistent colors and fonts
4. **Responsive Design**: Ensure plots work on different screen sizes
5. **Performance**: Optimize for large datasets
6. **Accessibility**: Use clear labels and colorblind-friendly palettes
7. **Export Quality**: Use appropriate resolution for different output formats

## Advantages
- **Interactivity**: Rich interactive features out of the box
- **Web-Ready**: Built for web deployment and sharing
- **Multiple Outputs**: Support for various export formats
- **Rich Features**: Extensive customization options
- **Integration**: Works well with Dash for dashboards
- **Community**: Active community and extensive documentation
- **Performance**: Efficient rendering for large datasets

## Limitations
- **Learning Curve**: Can be complex for advanced customizations
- **File Size**: Interactive HTML files can be large
- **Dependencies**: Requires additional libraries for some features
- **Static Export**: Limited options for static image export
- **Customization**: Some advanced features require JavaScript knowledge

## Related Libraries
- **Dash**: Web application framework for Plotly
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Static plotting library
- **Seaborn**: Statistical plotting built on matplotlib
- **Bokeh**: Alternative interactive plotting library 