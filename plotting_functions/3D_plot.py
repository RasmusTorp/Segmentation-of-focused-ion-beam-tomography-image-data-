from loadDataFunctions.load11t51center import data_load_tensors
from loadDataFunctions.loadMing import load_ming

import torch
import plotly.graph_objects as go

import numpy as np


X, y = data_load_tensors(folder_path="data/11t51center")

data = y.numpy()

# Define the coordinates
x, y, z = np.indices(data.shape)

# Create a 3D scatter plot
fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=data.flatten(),
    isomin=data.min(),
    isomax=data.max(),
    opacity=0.1, # Adjust the opacity for better visualization
    surface_count=20, # Number of isosurfaces
))

# Set the labels
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='3D Tensor Visualization'
)

# Show the plot
fig.show()