from typing import List, Optional, Tuple, Union, Dict
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def plot_filter_3d(filter_2d):
    """
    Visualize a 2D filter in 3D representation
    
    Parameters:
    filter_2d (numpy.ndarray): 2D filter/kernel to visualize
    """
    # Create coordinate grids
    x = np.arange(filter_2d.shape[1])
    y = np.arange(filter_2d.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(X, Y, filter_2d, 
                            cmap = "viridis",
                            linewidth=0, 
                            antialiased=False)
    
    # Customize the plot
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Filter Value')
    ax.set_title('3D Visualization of 2D Filter')
    
    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def plot_filter_3d_plotly(filter_2d):
    """
    Visualize a 2D filter in an interactive 3D Plotly representation
    Parameters:
    filter_2d (numpy.ndarray): 2D filter/kernel to visualize
    """
    # We need to flip the 2d-filter 
    filter_2d = np.flip(filter_2d, 1)
    x = np.arange(filter_2d.shape[1])
    y = np.arange(filter_2d.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create Plotly 3D surface plot
    fig = go.Figure(data=[go.Surface(z=filter_2d, x=X, y=Y, 
                                     colorscale='Viridis')])
    
    fig.update_layout(
        title='3D Visualization of 2D Filter',
        scene = dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Filter Value'
        ),
        width=800,
        height=600
    )

    #st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig)