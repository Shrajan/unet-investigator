# Note from the author Shrajan.
# If you reading this, then I am sorry for what you are about to see. 
# This piece of code is absolutely basic, crudely written, unoptimized, and straight up madness.
# It is possible that some things may not work for you or it might crash your device.
# So I urge you to stop here, and go no further. PLEASE STOP!!!
# Even after reading the above warnings, if you proceed, then I am not to be blamed.
# DO THIS AT YOUR OWN PERIL. 

import streamlit as st
import torch, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict
from matplotlib.colors import LinearSegmentedColormap
from copy import copy
from PIL import Image
from matplotlib import cm
import plotly.graph_objs as go

# Assuming you have these utility functions in separate files
from nnunet_helper import nnUNetHelper
from preprocessing_utils import preprocess_ct_slice

st.set_page_config(
            page_title="U-Net Explorer", 
            page_icon="🩻",
            #layout="wide"
)

# Some colors
st.html(
        '''
        <style>
        hr {
            border-color: red;
        }
        </style>
        '''
)

def get_user_friendly_names(actual_layer_names: List, num_unet_stages: int, nnunet_format: bool =True) -> Dict:
    layer_name_mapping_dict = {}
    #print(actual_layer_names)
    
    if nnunet_format:
        num_encoders = num_unet_stages
        num_decoders = num_unet_stages - 1
        num_deep_supervision = num_decoders
        try:
            for layer_name in actual_layer_names:
                str_split_list = layer_name.split(".")
                
                
                if 'seg_layers' in layer_name:
                    if str_split_list[-1] == str(num_deep_supervision-1):
                        user_friendly_name = "Final Layer - Prediction"
                    else:
                        user_friendly_name = f"{str_split_list[0].title()} {num_decoders - int(str_split_list[-1])} - Deep Supervision" 
                
                elif "decoder" in layer_name.lower():
                    if 'dws_conv' in layer_name:
                        dws_dict = {'0': 'Depth-wise', '1': 'Point-wise'}
                        user_friendly_name = f"{str_split_list[0].title()} {num_decoders - int(str_split_list[2])} - {dws_dict[str_split_list[-1]]} Convolution {int(str_split_list[-3])+1}" 
                    else:
                        user_friendly_name = f"{str_split_list[0].title()} {num_decoders - int(str_split_list[2])} - Normal/Patch Convolution {int(str_split_list[-3])+1}" 
                    
                elif "encoder" in layer_name.lower():
                    if int(str_split_list[2]) == num_encoders -1:
                        if 'dws_conv' in layer_name:
                            dws_dict = {'0': 'Depth-wise', '1': 'Point-wise'}
                            user_friendly_name = f"Middle/Bottleneck - {dws_dict[str_split_list[-1]]} Convolution {int(str_split_list[-3])+1}" 
                        else:
                            user_friendly_name = f"Middle/Bottleneck - Normal/Patch Convolution {int(str_split_list[-3])+1}" 
                    else:
                        if 'dws_conv' in layer_name:
                            dws_dict = {'0': 'Depth-wise', '1': 'Point-wise'}
                            user_friendly_name = f"{str_split_list[0].title()} {int(str_split_list[2])+1} - {dws_dict[str_split_list[-1]]} Convolution {int(str_split_list[-3])+1}" 
                        else:
                            user_friendly_name = f"{str_split_list[0].title()} {int(str_split_list[2])+1} - Normal/Patch Convolution {int(str_split_list[-3])+1}" 
                
                else:
                    raise Exception("Could not find a matching prefix name in the U-Net model.")
                
                layer_name_mapping_dict[layer_name] = user_friendly_name
            return layer_name_mapping_dict
        except:
            raise Exception("Could not map user friendly names.")
    else:
        raise Exception("Only nnUNet framework is supported for now.")

def extract_slice(img_array):
    """
    Extract a representative 2D slice from multi-dimensional arrays
    
    """
    # Handle different dimensional inputs
    if img_array.ndim == 2:
        return img_array
    elif img_array.ndim == 3:
        # Select middle slice
        slice_selection = img_array.shape[2] // 2
        return img_array[:, :, slice_selection]
    elif img_array.ndim == 4:
        # For multi-channel 3D images, select middle slice of first channel
        slice_selection = img_array.shape[2] // 2
        return img_array[:, :, slice_selection, 0]
    else:
        raise ValueError(f"Unsupported image dimensions: {img_array.ndim}")

def load_medical_image(uploaded_file: Union[str, np.ndarray, None], 
                        file_type: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    Comprehensive medical image loader supporting multiple input types
    
    Args:
        uploaded_file: Input image source (file path, numpy array, Streamlit uploaded file)
        file_type: Optional explicit file type specification
    
    Returns:
        Tuple of (preprocessed_slice, original_slice, metadata)
    """

    # Default error handling
    if uploaded_file is None:
        st.info("""
        ### 📸 Image Upload Guidance
        Supported Input Types:
        - Streamlit file uploader
        - NumPy array
        - File path to medical image
        - Supports 2D/3D/4D arrays
        - Recommended: Grayscale medical images
        """)
        return None, None, None

    try:
        # Handle different input types
        if isinstance(uploaded_file, np.ndarray):
            # Direct NumPy array input
            original_slice = extract_slice(uploaded_file)
            metadata = {
                'input_type': 'numpy_array',
                'original_shape': uploaded_file.shape,
                'data_type': str(uploaded_file.dtype)
            }
        
        elif isinstance(uploaded_file, str):
            # File path input
            file_extension = uploaded_file.split('.')[-1].lower()
            
            if file_extension in ['npy']:
                # NumPy .npy file
                original_slice = extract_slice(np.load(uploaded_file))
                metadata = {
                    'input_type': 'numpy_file',
                    'file_path': uploaded_file
                }
            
            elif file_extension in ['nii', 'nii.gz']:
                # NIfTI file
                nifti_img = nib.load(uploaded_file)
                img_array = nifti_img.get_fdata()
                original_slice = extract_slice(img_array)
                
                metadata = {
                    'input_type': 'nifti_file',
                    'file_path': uploaded_file,
                    'original_dimensions': img_array.shape,
                    'affine_matrix': nifti_img.affine
                }
            
            else:
                # Other image formats
                original_slice = plt.imread(uploaded_file)
                if original_slice.ndim == 3:
                    original_slice = original_slice[:, :, 0]
                
                metadata = {
                    'input_type': 'image_file',
                    'file_path': uploaded_file
                }
        
        elif hasattr(uploaded_file, 'read'):
            # Streamlit uploaded file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['npy']:
                # NumPy .npy file from upload
                original_slice = extract_slice(np.load(uploaded_file))
                metadata = {
                    'input_type': 'uploaded_numpy_file',
                    'filename': uploaded_file.name
                }
            
            elif file_extension in ['nii', 'nii.gz']:
                # NIfTI file from upload
                nifti_img = nib.load(uploaded_file)
                img_array = nifti_img.get_fdata()
                original_slice = extract_slice(img_array)
                
                metadata = {
                    'input_type': 'uploaded_nifti',
                    'filename': uploaded_file.name,
                    'original_dimensions': img_array.shape
                }
            
            else:
                # Standard image formats
                original_slice = plt.imread(uploaded_file)
                if original_slice.ndim == 3:
                    original_slice = original_slice[:, :, 0]
                
                metadata = {
                    'input_type': 'uploaded_image',
                    'filename': uploaded_file.name
                }
                
        else:
            st.error("Unsupported input type")
            return None, None, None
        
        # Ensure slice is float for preprocessing
        original_slice = original_slice.astype(np.float32)
        
        # Add general metadata
        metadata.update({
            'slice_shape': original_slice.shape,
            'min_value': np.min(original_slice),
            'max_value': np.max(original_slice),
            'mean_value': np.mean(original_slice),
            'std_value': np.std(original_slice)
        })
        
        # Preprocessing
        preprocessed_slice = preprocess_ct_slice(original_slice)
        
        return preprocessed_slice, original_slice, metadata

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None


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
                            #cmap=cm.coolwarm,  # Color map 
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

class AdvancedCNNVisualizer:
    def __init__(self, model):
        self.model = model
        
    def load_model_weights(self, network_weights: Dict, 
                           nnunet_format: bool =True):
        
        if nnunet_format:
            self.model = nnUNetHelper.nnunet_load_weights(network=self.model, 
                                                          trained_network_weights=network_weights) 
        else:
            raise Exception("Only nnUNet framework is supported for now.")
    
    def get_feature_maps(self, input_tensor: torch.Tensor, layer_name: str, 
                          threshold: float = 0.0) -> Optional[torch.Tensor]:
        """
        Extract and threshold feature maps
        
        Args:
            input_tensor (torch.Tensor): Input image tensor
            layer_name (str): Layer to extract features from
            threshold (float): Threshold to apply to feature maps
        
        Returns:
            Thresholded feature maps or None
        """
        feature_maps = []
        try:
            def hook_fn(module, input, output):
                feature_maps.append(output.detach())
            
            layer = dict(self.model.named_modules())[layer_name]
            handle = layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                self.model(input_tensor)
            
            handle.remove()
            return feature_maps[0]
        except Exception as e:
            st.error(f"Error extracting feature maps: {e}")
            return None

    def advanced_kernel_analysis_1d(self, 
                                  layer_name: str, 
                                  cmap: str = 'viridis', 
                                  normalize: bool = True,
                                  num_columns: int = 1,
                                  figsize: Tuple =(12, 12),
                                  fontsize: int =8,
                                  apply_threshold: bool = False,
                                  threshold_value: float = 0.0,
                                  xticks_step:int = 1) -> Tuple[plt.Figure, dict]:
        """
        Advanced kernel visualization with detailed analysis
        
        Args:
            layer_name (str): Name of the layer to analyze
            cmap (str): Colormap for visualization
            normalize (bool): Whether to normalize kernel values
        
        Returns:
            Matplotlib figure and kernel statistics
        """
        layer = dict(self.model.named_modules())[layer_name]
        
        if not hasattr(layer, 'weight'):
            st.error(f"Layer {layer_name} does not have weights")
            return None, {}
        
        kernels = layer.weight.detach().cpu().numpy()
        kernels_shape = kernels.shape
        
        n_kernels = kernels_shape[0]
        grid_size = int(np.ceil(n_kernels/num_columns))
        
        fig, axes = plt.subplots(grid_size, num_columns, figsize=figsize)
        axes = axes.ravel() if grid_size > 1 else [axes]
        
        # Kernel statistics
        kernel_stats = {
            'total_kernels': n_kernels,
            'kernel_shape': kernels.shape[1:],
            'kernel_details': []
        }
        
        for i in range(kernels_shape[0]):
            kernel = kernels[i].flatten()
            
            if apply_threshold:
                kernel = np.where(np.abs(kernel) > threshold_value, kernel, 0)
            
            if normalize:
                kernel = (kernel - np.mean(kernel)) / np.std(kernel)
            
            # Kernel-level statistics
            kernel_info = {
                'kernel id': i+1,
                'sum': np.sum(kernel),
                'mean': np.mean(kernel),
                'std': np.std(kernel),
                'max': np.max(kernel),
                'min': np.min(kernel)
            }
            kernel_stats['kernel_details'].append(kernel_info)
            
            axes[i].plot(kernel, linewidth=1.0)
            axes[i].set_xticks(np.arange(len(kernel), step=xticks_step), np.arange(1, len(kernel)+1, step=xticks_step))
            axes[i].grid()   
            #axes[i].axis('off')
            
            # Annotate with key statistics
            axes[i].set_title(
                f'Kernel {i+1}\n'
                f'Sum: {kernel_info["sum"]:.3f}', 
                fontsize=fontsize
            )
        
        # Hide unused subplots
        for j in range(n_kernels, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        return fig, kernel_stats
    
    def advanced_kernel_analysis_Nd(self, 
                                  layer_name: str, 
                                  cmap: str = 'viridis', 
                                  normalize: bool = True,
                                  figsize: Tuple =(12, 12),
                                  fontsize: int =8) -> Tuple[plt.Figure, dict]:
        """
        Advanced kernel visualization with detailed analysis
        
        Args:
            layer_name (str): Name of the layer to analyze
            cmap (str): Colormap for visualization
            normalize (bool): Whether to normalize kernel values
        
        Returns:
            Matplotlib figure and kernel statistics
        """
        layer = dict(self.model.named_modules())[layer_name]
        
        if not hasattr(layer, 'weight'):
            st.error(f"Layer {layer_name} does not have weights")
            return None, {}
        
        kernels = layer.weight.detach().cpu().numpy()
        kernels_shape = kernels.shape
        
        n_kernels = kernels_shape[0] * kernels_shape[1]
        grid_size = int(np.ceil(np.sqrt(n_kernels)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.ravel() if grid_size > 1 else [axes]
        
        # Kernel statistics
        kernel_stats = {
            'total_kernels': n_kernels,
            'kernel_shape': kernels.shape[2:],
            'kernel_details': []
        }
        
        i=0 # Counter 
        for m in range(kernels_shape[0]):
            for n in range(kernels_shape[1]):
                kernel = kernels[m, n]
                if normalize:
                    kernel = (kernel - np.mean(kernel)) / np.std(kernel)
                
                # Kernel-level statistics
                kernel_info = {
                    'kernel id': i+1,
                    'sum': np.sum(kernel),
                    'mean': np.mean(kernel),
                    'std': np.std(kernel),
                    'max': np.max(kernel),
                    'min': np.min(kernel)
                }
                kernel_stats['kernel_details'].append(kernel_info)
                
                axes[i].imshow(kernel, cmap=cmap)
                axes[i].axis('off')
                
                # Annotate with key statistics
                axes[i].set_title(
                    f'Kernel {i+1}\n'
                    f'Sum: {kernel_info["sum"]:.3f}', 
                    fontsize=fontsize
                )
                i+=1
        
        # Hide unused subplots
        for j in range(n_kernels, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        return fig, kernel_stats

class MedicalImageExplorer:
    @staticmethod
    def add_visualization_context():
        """Provides in-depth contextual explanations for technical visualizations"""
        st.sidebar.markdown("## 🧠 Visualization Context")
        
        with st.sidebar.expander("🔬 Kernel Visualization Explained"):
            st.markdown("""
            #### What are Kernels?
            - **Neural Network Building Blocks**: Kernels are small matrices that act as feature detectors
            - **Learning Filters**: Each kernel learns to recognize specific patterns in images
            
            #### Kernel Interpretation
            - **Bright Regions**: Strong feature responses
            - **Dark Regions**: Weak or negative feature responses
            - **Variation**: Indicates complexity of learned features
            
            #### Common Kernel Patterns
            1. **Edge Detectors**: 
            - Highlight boundaries and transitions
            - Crucial for medical image segmentation
            2. **Texture Analyzers**:
            - Detect tissue patterns
            - Important for distinguishing between healthy and abnormal regions
            3. **Shape Recognizers**:
            - Identify specific geometric structures
            - Vital for organ or tumor identification
            """)
        
        with st.sidebar.expander("🌈 Feature Map Deep Dive"):
            st.markdown("""
            #### What are Feature Maps?
            - **Activation Visualization**: Shows how different image regions activate neural network layers
            - **Pattern Detection**: Reveals what patterns the network recognizes
            
            #### Feature Map Hierarchy
            1. **Early Layers** (First few layers):
            - Detect basic features like edges, corners
            - Simple, low-level patterns
            
            2. **Middle Layers**:
            - Combine basic features
            - Recognize more complex shapes
            - Identify texture and local structures
            
            3. **Deep Layers**:
            - Capture high-level semantic information
            - Detect complex, abstract patterns
            - Critical for medical image interpretation
            
            #### Medical Imaging Context
            - **Tumor Detection**: Feature maps can highlight potential abnormal regions
            - **Organ Segmentation**: Visualize how network distinguishes different tissues
            - **Diagnostic Support**: Provide insights into neural network's decision-making
            """)
        
    @staticmethod
    def main():
        
        st.title('What happens under the hood of a U-Net.')
        
        
        # Enhanced sidebar with comprehensive guidance
        st.sidebar.markdown("""
        ## 🔬 Medical Imaging & Neural Network Visualization
        
        Explore medical images and understand neural network interpretations.
        
        ### Features
        1. Neural Network Visualization - CNN Kernels 
        2. Advanced Image Loading and Feature Map Analysis
        """)
        st.sidebar.divider()  # 👈 Draws a horizontal rule
        
        # Neural Network Visualization Mode
        st.sidebar.header('Load Model Checkpoint')
        uploaded_model = st.sidebar.file_uploader(
            "Upload PyTorch checkpoint file", 
            type=['pth', 'pt'],
            help="Upload a checkpoint file for U-Net model in the nnUNet format"
        )
        st.sidebar.divider()  # 👈 Draws a horizontal rule
        
        ##########################################
        # Read Input - Feature Map Visualization
        ##########################################
        st.sidebar.header("Upload Input Image")
        # Image loading section from first script
        input_type = st.sidebar.radio(
            "Select Input Type", 
            ["Single Slice as NumPy Array", "Medical Image (incomplete implementation)"]
        )
        
        # Input based on selected type
        if input_type == "Medical Image":
            # Image upload with detailed guidance
            uploaded_image = st.sidebar.file_uploader(
                "Upload Medical Image", 
                type=['nii', 'nii.gz', 'png', 'jpg', 'jpeg', '.npy'],
                help="""
                Upload a medical image for feature map visualization
                - Recommended: Single-channel grayscale images
                - Supports NIfTI and standard image formats
                """
            )
        
        elif input_type == "Single Slice as NumPy Array":
            # NumPy file upload
            uploaded_image = st.sidebar.file_uploader(
                "Upload .npy File", 
                type=['npy'],
                help="Upload a NumPy .npy file"
            )
        else:
            uploaded_image = None
        
        st.sidebar.divider()  # 👈 Draws a horizontal rule
        
        if uploaded_model is None:
            st.markdown("""
                ### 🚀 To get started
                - Please upload a pre-trained U-Net model under the `Load Model Checkpoint` tab in the sidebar.
                - At this moment, we are supporting U-Net models created using the ***nnUNet V2*** framework.
                - It is preferred to use a ***2D*** configuration, as 3D is difficult to visualize and understand. 
                - We are using this to understand the inner workings of ***Depth-wise Separable*** convolutions.
                """)
            st.divider()  # 👈 Draws a horizontal rule

        elif uploaded_model is not None:
            # Model loading
            try:
                network, network_weights, network_configuration, network_init_kwargs = nnUNetHelper.nnunet_model_loading(savedCheckpointPath=uploaded_model)
                network = nnUNetHelper.nnunet_load_weights(network=network, trained_network_weights=network_weights)
            except Exception as e:
                st.error(f"Model loading failed: {e}")
                network = None
            
            if network is not None:
                st.markdown("""
                ### ☑️ Model loaded successfully
                - A complete list of ***convolution layers*** of the U-Net can be found in the sidebar.
                - You can choose ***single*** or ***multiple*** layers as needed, and their ***kernels*** will be displayed below.
                - Kernels of ***Normal*** or ***Depth-wise convolutions*** will be displayed as matrices.
                - ***Point-wise convolutions*** will be shown as plots. 
                - Respective options of different layers can be modified as required.
                """)
                st.write(" . "*98) # Found this value using trial and error for my machine - it is not dynamic because for some reason streamlit does not give page width. And I was lazy to search further.
                st.markdown("""
                ### Feature maps visualization
                - If you wish to visualize the feature maps, please upload a file under the `Upload Input Image` tab in the sidebar. 
                - You can visualize various feature maps: ***input image slice***, ***model prediction***, and ***layer outputs***.
                """)
                st.divider()  # 👈 Draws a horizontal rule
                
                # DONE: Message - model loaded successful.
                # TODO: Architecture - image of the trained model - Hidden layer or PyTorchViz

                
                # Process and display image
                if uploaded_image is not None:
                    input_image, original_slice, metadata = load_medical_image(uploaded_image)
                    
                    if input_image is not None:
                        
                        # Visualization of original image
                        st.markdown("### Feature Maps of the uploaded Image")
                        #st.write("Image Metadata:", metadata)
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.title("Original Image")
                        plt.imshow(input_image, cmap='gray')
                        plt.subplot(1, 2, 2)
                        plt.title("Model Prediction")
                        colors = ["k", "w", "r", "g", "b", "m", "c", "y"]
                        cmap_prediction = LinearSegmentedColormap.from_list("mycmap", colors)
                        plt.imshow(nnUNetHelper.predict_on_2d(network=network, input_array=input_image), 
                                   cmap=cmap_prediction)
                        st.pyplot(plt.gcf())
                        st.divider()  # 👈 Draws a horizontal rule
                else:
                    input_image = None
                
                ##########################################
                # Kernel Visualization
                ##########################################
                
                # DONE: Segregate - for each layer make a new block.
                # TODO: Segregate - Relu and Normalization layers.
                # TODO: Layer names - change the names of the layers based on their position. 
                #                   - use the architecture as a template.
                # TODO: Options - To choose single, multiple or all kernels in a layer.
                # TODO: Resize - resize shape, grids, height, width, font of figures.
                
                #st.header('Kernel Visualization')
                # Enhanced kernel analysis explanation
                st.sidebar.header('🔍 Kernel Visualization')
                st.sidebar.markdown("""
                - **Kernels** are learned filters that detect specific patterns in images
                - Each kernel represents a unique feature detector
                - Brighter/darker regions indicate stronger feature responses
                ### List of available convolution layers
                """)
                
                visualizer = AdvancedCNNVisualizer(network)
                
                # Layer and visualization mode selection
                all_conv_layers = [
                    name for name, module in network.named_modules() 
                    if isinstance(module, torch.nn.Conv2d)
                ]
                required_layers = copy(all_conv_layers)
                # Remove the deep supervision layers, except final prediction layer.
                for a_conv_layer_name in all_conv_layers:
                    if 'seg_layers' in a_conv_layer_name:
                        str_split_list = a_conv_layer_name.split(".")
                        if str_split_list[-1] != str(network_init_kwargs["n_stages"]-2):
                            required_layers.remove(a_conv_layer_name)
                
                layer_mapping_dict = get_user_friendly_names(actual_layer_names=required_layers, 
                                                             num_unet_stages=network_init_kwargs["n_stages"])
                
                for idx, a_layer in enumerate(required_layers):
                    
                    # Kernel analysis controls
                    layer_selection_button = st.sidebar.checkbox(
                        #f'Visualize kernels and feature maps of {layer_mapping_dict[a_layer]}', 
                        f'{layer_mapping_dict[a_layer]}', 
                        value=False,
                    )

                    if layer_selection_button:
                        
                        # Name of selected layer
                        #st.subheader(f'Displaying kernels of layer: ***{layer_mapping_dict[a_layer]}***')
                        name_required_weight = str(a_layer +".weight")
                        required_weight_shape = network_weights[name_required_weight].shape
                        num_layer_kernels =required_weight_shape[0]
                        
                        st.markdown(f"""
                        #### Displaying kernels of layer :red[{layer_mapping_dict[a_layer]}]
                        This layer has :blue-background[{num_layer_kernels}] kernels. 
                        You can choose to display :orange[all], and/or enter below your desired kernel ID.
                        """)
                        
                        # Display all kernels controls
                        display_all_kernels = st.checkbox(
                            f'Display :orange[all] kernels of layer: {layer_mapping_dict[a_layer]}', 
                            value=False,
                            #help="Scales kernel values to a consistent range for better comparison"
                        )
                        
                        if display_all_kernels:
                            st.write("##### Plots with Original Values/Distribution")
                            
                            # Some variables to prevent chaos.
                            kernel_fig = None
                            threshold_all_plots = False
                            
                            with st.expander(f"Control panel of layer: {layer_mapping_dict[a_layer]}"):
                                if required_weight_shape[-1] == 1:
                                    # This does not matter.
                                    cmap_kernels = 'viridis'
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # For setting the height and width of the plots.
                                        layer_plot_height = st.slider(f"Plot Height for layer: {layer_mapping_dict[a_layer]}", 1, 100, 60)
                                        layer_plot_width = st.slider(f"Plot Width for layer: {layer_mapping_dict[a_layer]}", 1, 50, 10)
                                        layer_figsize = (layer_plot_width, layer_plot_height)
                                        
                                        # Threshold controls
                                        threshold_all_plots = st.checkbox(
                                            f'Apply threshold and create new plot for layer: {layer_mapping_dict[a_layer]}', 
                                            value=False,
                                            help="Thresholds all the values of the layer"
                                        )
                                        
                                    with col2:
                                        # Plot columns
                                        layer_plot_columns = st.slider(f"Number of columns for layer: {layer_mapping_dict[a_layer]}", 1, num_layer_kernels, 1)
                                        
                                        # Plot font size
                                        layer_plot_font = st.slider(f"Font-Size for layer: {layer_mapping_dict[a_layer]}", 1, 25, 12)

                                        # Kernel analysis controls
                                        normalize_kernels = st.checkbox(
                                            f'Normalize kernel values for layer: {layer_mapping_dict[a_layer]}', 
                                            value=False,
                                            help="Scales kernel values to a consistent range for better comparison"
                                        )
                                        
                                        xticks_step_full_plot_str = st.text_input(f"Enter the number (positive integer) of steps between x-ticks for {layer_mapping_dict[a_layer]}.", 
                                                            "1")
                                        
                                        try:
                                            xticks_step_full_plot_int = int(xticks_step_full_plot_str)
                                            if xticks_step_full_plot_int < 1 or xticks_step_full_plot_int > required_weight_shape[1]:
                                                st.write(f"The current steps value is invalid and we will use a step-size of 1. Please enter a positive integer value between 1-{required_weight_shape[1]}.",)
                                                xticks_step_full_plot_int = 1
                                        except:
                                            st.write(f"The current steps value is invalid and we will use a step-size of 1. Please enter a positive integer value between 1-{required_weight_shape[1]}.",)
                                            xticks_step_full_plot_int = 1
                                        
                                    # Generate kernel visualization
                                    kernel_fig, kernel_stats = visualizer.advanced_kernel_analysis_1d(
                                        layer_name=a_layer, 
                                        normalize=normalize_kernels,
                                        num_columns=layer_plot_columns,
                                        figsize=layer_figsize,
                                        fontsize=layer_plot_font,
                                        xticks_step=xticks_step_full_plot_int
                                    )
                                     
                                    
                                else:
                                    layer_figsize = (12,12)
                                    col1, col2 = st.columns(2)
                            
                                    with col1:
                                        # Common colormap selection
                                        cmap_kernels = st.selectbox(
                                            f'Colormap for all kernels of layer: {layer_mapping_dict[a_layer]}', 
                                            ['viridis', 'gray', 'plasma', 'inferno', 'magma', 'cividis', 'jet']
                                        )
                                    
                                    with col2:
                                        # Plot font size
                                        layer_plot_font = st.slider(f"Font-Size for layer: {layer_mapping_dict[a_layer]}", 1, 25, 12)

                                    # Kernel analysis controls
                                    normalize_kernels = st.checkbox(
                                        f'Normalize kernel values for layer: {layer_mapping_dict[a_layer]}', 
                                        value=False,
                                        help="Scales kernel values to a consistent range for better comparison"
                                    )
                                
                                    # Generate kernel visualization
                                    kernel_fig, kernel_stats = visualizer.advanced_kernel_analysis_Nd(
                                        layer_name=a_layer, 
                                        cmap=cmap_kernels, 
                                        normalize=normalize_kernels,
                                        figsize=layer_figsize,
                                        fontsize=layer_plot_font
                                    )
                                    
                            # Plot the original kernel values.
                            if kernel_fig is not None:
                                st.pyplot(kernel_fig)
                                
                                # Display kernel statistics
                                with st.expander(f'Kernel Statistics of original plots of layer: {layer_mapping_dict[a_layer]}'):
                                    st.write(f"Total Kernels: {kernel_stats['total_kernels']}")
                                    st.write(f"Kernel Shape: {kernel_stats['kernel_shape']}")
                                    
                                    # Detailed kernel stats
                                    stats_df = pd.DataFrame(kernel_stats['kernel_details'])
                                    st.dataframe(stats_df, hide_index=True)
                            
                                if threshold_all_plots:
                                    st.write(" . "*98) # Found this value using trial and error for my machine - it is not dynamic because for some reason streamlit does not give page width. And I was lazy to search further.
                                    st.write("##### Plots with Threshold Values")
                                    threhold_value_str = st.text_input(f"Enter the threshold value for: {layer_mapping_dict[a_layer]}",
                                                    "0.0")
                                    try:
                                        threhold_value_num = float(threhold_value_str)
                                        # Generate kernel visualization
                                        kernel_after_threshold_fig, kernel_after_threshold__stats = visualizer.advanced_kernel_analysis_1d(
                                            layer_name=a_layer, 
                                            normalize=normalize_kernels,
                                            num_columns=layer_plot_columns,
                                            figsize=layer_figsize,
                                            fontsize=layer_plot_font,
                                            apply_threshold=threshold_all_plots,
                                            threshold_value=threhold_value_num
                                        )
                                        st.pyplot(kernel_after_threshold_fig)
                                         # Display kernel statistics
                                        with st.expander(f'Kernel Statistics of threshold plots of layer: {layer_mapping_dict[a_layer]}'):
                                            st.write(f"Total Kernels: {kernel_stats['total_kernels']}")
                                            st.write(f"Kernel Shape: {kernel_stats['kernel_shape']}")
                                            
                                            # Detailed kernel stats
                                            stats_df = pd.DataFrame(kernel_stats['kernel_details'])
                                            st.dataframe(stats_df, hide_index=True)
                                        
                                    except:
                                        st.write(f"The current threshold value is invalid. Please enter a number and re-try, or disable ***threshold*** in the control panel.",)
                                            
                        
                        st.write(" . "*98) # Found this value using trial and error for my machine - it is not dynamic because for some reason streamlit does not give page width. And I was lazy to search further.
                        required_kernel_ids = st.text_input(f"Enter the required kernel ID for: {layer_mapping_dict[a_layer]}. If you want to display multiple kernels, please separate ID values with commas (example: ***3*** or ***1,2,3*** or ***4,8,12*** or ***10-15*** or ***9,16-20*** or ***4-8,16-20*** or ***all*** or ***none***).", 
                                                            "none")      
                        
                        # To save some hassle.
                        required_kernel_ids_list = None
                        try:
                            if required_kernel_ids == "none":
                                pass
                            else:
                                if required_kernel_ids.lower() == "all":
                                    required_kernel_ids_list = [x for x in range(1,num_layer_kernels+1)]
                                    st.write(f"All kernels are selected, and their attributes will be displayed individually below.")  
                                elif "-" in required_kernel_ids and "," in required_kernel_ids:
                                    required_kernel_ids_list = []
                                    comma_split_list = required_kernel_ids.split(",")
                                    for list_val in comma_split_list:
                                        if "-" in list_val:
                                            hypen_split_list = list_val.split("-")
                                            if len(hypen_split_list) == 2:
                                                required_kernel_ids_list.extend([hypen_val for hypen_val in range(int(hypen_split_list[0]), int(hypen_split_list[1])+1)])
                                            else:
                                                st.write(f"The current value is invalid. Please enter valid kernel ID(s) in integer format and re-try.",)
                                                required_kernel_ids_list = None
                                                break
                                        else:
                                            required_kernel_ids_list.append(int(list_val))
                                        
                                elif "-" in required_kernel_ids and "," not in required_kernel_ids:
                                    required_kernel_ids_list=[]
                                    hypen_split_list = required_kernel_ids.split("-")
                                    if len(hypen_split_list) == 2:
                                        required_kernel_ids_list.extend([hypen_val for hypen_val in range(int(hypen_split_list[0]), int(hypen_split_list[1])+1)])
                                    else:
                                        st.write(f"The current value is invalid. Please enter valid kernel ID(s) in integer format and re-try.",)
                                        required_kernel_ids_list = None
                                        break
                                else:
                                    required_kernel_ids_list = [int(x) for x in required_kernel_ids.split(",")]
                                    
                                if any( x < 1 or x > num_layer_kernels for x in required_kernel_ids_list):
                                    st.write(f"The current value is invalid. Please choose integer values between 1-{num_layer_kernels}.",)
                                    required_kernel_ids_list = None
                                else:  
                                    required_kernel_ids_list = sorted(set(required_kernel_ids_list))
                                    st.write(f"The selected kernel(s) are {required_kernel_ids_list}, and their attributes will be displayed individually below.")        
                        except:
                            st.write(f"The current value is invalid. Please enter valid kernel ID(s) in integer format and re-try.",)
                            required_kernel_ids_list = None
                            
                        
                        ##########################################
                        # Individual Feature Map Visualization
                        ##########################################
                        
                        # TODO: Segregate - for each kernel make a new block.
                        # TODO: Threshold - ability to modify all values or individual values.
                        #                 - Single slider for all values or individual sliders for individual values.
                        #                 - feature maps of each new modification.
                        # TODO: Model state - create new model states for each modified layer.
                        # TODO: Saving model - Ability to save modified weights.
                        #                    - Give options for each layer.
                        #                    - Default values for unchanged layers.
        
                        if required_kernel_ids_list is not None and input_image is None:
                            st.write(f"Please upload the an input image to visualize the feature maps of the kernels.")
                            
                        elif required_kernel_ids_list is not None and input_image is not None:
                            if "dws_conv.0" in a_layer or "seg_layers" in a_layer:
                                apply_norm_and_act = False
                                separate_norm_and_act = "No"
                                feature_maps_of_actlayer = None
                                st.write(f"There are no normalization or activation layers immediately after this layer.")
                            else:
                                # Normalization and Activation controls
                                apply_norm_and_act = st.checkbox(
                                    f'Show feature maps after applying normalization and activation for selected kernels of layer: {layer_mapping_dict[a_layer]}', 
                                    value=False,
                                    help="Applies normalization and activation on the convolution output."
                                )
                                
                                if apply_norm_and_act:
                                    separate_norm_and_act = st.radio(f"Display individual feature maps of normalization and activation for selected kernels of layer: {layer_mapping_dict[a_layer]}.", 
                                                                ["No", "Yes"]
                                                                )
                            
                            if required_weight_shape[-1] != 1:
                                # Show Nd kernels as 3D plots.
                                show_3d_plots = st.checkbox(
                                    f'Display 2D kernels in 3D for selected kernels of layer: {layer_mapping_dict[a_layer]} (this is a memory intensive step, so please limit the number of kernels you wish to see).', 
                                    value=False,
                                    help="Plot 3D plots for the chosen 2D kernels."
                                )
                            else:
                                show_3d_plots = False
                            
                            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                            input_image_tensor = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).to(device)
                            feature_maps_of_conv_layer = visualizer.get_feature_maps(input_tensor=input_image_tensor,
                                                        layer_name=a_layer)                         
                            
                            if apply_norm_and_act and feature_maps_of_conv_layer is not None:
                                layer_name_act = copy(a_layer) # Because I am being pedantic.
                                layer_name_norm = copy(a_layer) # And also sleep deprived.
                                
                                layer_name_act = layer_name_act.replace("conv", "nonlin") if a_layer == "encoder.stages.0.0.conv" else layer_name_act.replace("dws_conv.1", "nonlin")
                                layer_name_norm = layer_name_norm.replace("conv", "norm") if a_layer == "encoder.stages.0.0.conv" else layer_name_norm.replace("dws_conv.1", "norm")
                                
                                feature_maps_of_actlayer = visualizer.get_feature_maps(input_tensor=input_image_tensor,
                                                            layer_name=layer_name_act) 
                                
                                imageFilePath_NormAct = "helpers/images/Norm_plus_activation.png" 
                                image_NormAct = np.array(Image.open(imageFilePath_NormAct), dtype=np.uint8)

                                if separate_norm_and_act == "Yes":                                
                                    #feature_maps_of_normlayer = visualizer.get_feature_maps(input_tensor=input_image_tensor,
                                    #                            layer_name=layer_name_norm)
                                    
                                    # Why am I manually doing normalization you ask? 
                                    # Well activation was done 'inplace', so both output norm and leakyrelu give the same answer. 
                                    try:
                                        layer_norm_weight_original_kernel = network_weights[f"{layer_name_norm}.weight"]
                                        layer_norm_bias_original_kernel = network_weights[f"{layer_name_norm}.bias"]
                                        feature_maps_of_normlayer = torch.nn.functional.instance_norm(input=feature_maps_of_conv_layer,
                                                                                                weight=layer_norm_weight_original_kernel,
                                                                                                bias=layer_norm_bias_original_kernel) 
                                    except:
                                        feature_maps_of_normlayer = torch.nn.functional.instance_norm(input=feature_maps_of_conv_layer)
                                        
                                    imageFilePath_Norm = "helpers/images/Normalization.png" 
                                    image_Norm = np.array(Image.open(imageFilePath_Norm), dtype=np.uint8)
                                    
                                    imageFilePath_Act = "helpers/images/Activation.png" 
                                    image_Act = np.array(Image.open(imageFilePath_Act), dtype=np.uint8)
                                    
                                    
                            for kernel_id in required_kernel_ids_list:
                                
                                st.write(" .. "*63) # Found this value using trial and error for my machine - it is not dynamic because for some reason streamlit does not give page width. And I was lazy to search further.
                                st.write(f"##### Kernel #{kernel_id} of layer: {layer_mapping_dict[a_layer]}")
                                plt.figure(figsize=(10, 5))
                                
                                plt.subplot(1, 2, 1)
                                if required_weight_shape[-1] == 1:
                                    kernel_pw_plot = network_weights[name_required_weight][kernel_id-1].cpu().numpy().flatten()                        
                                    plt.plot(kernel_pw_plot, linewidth=1.0)
                                    xticks_step = math.ceil(kernel_pw_plot.shape[0]/16) # Why 16? Because 16 ticks fit in the plot. Deal with it!
                                    plt.xticks(np.arange(len(kernel_pw_plot),step=xticks_step), np.arange(1, len(kernel_pw_plot)+1, step=xticks_step))  
                                    plt.grid()  
                                    plt.title(f"Original Kernel\n"
                                          f"Sum = {np.sum(kernel_pw_plot):.3f}")
                                else:
                                    kernel_dw_plot = network_weights[name_required_weight][kernel_id-1,0].cpu().numpy()
                                    plt.imshow(kernel_dw_plot, 
                                           cmap='viridis')
                                    
                                    # Add text annotations for each cell
                                    for i in range(kernel_dw_plot.shape[0]):
                                        for j in range(kernel_dw_plot.shape[1]):
                                            plt.text(j, i, str(round(kernel_dw_plot[i, j],3)), 
                                                    ha='center', va='center', fontsize = 'small',
                                                    color='white' if kernel_dw_plot[i, j] < np.mean(kernel_dw_plot) else 'black')
                                    plt.axis("off") 
                                    plt.title(f"Original Kernel\n"
                                          f"Sum = {np.sum(kernel_dw_plot):.3f}")
                                    
                                plt.subplot(1, 2, 2)
                                plt.title("Feature map using original Kernel\n"
                                          f"Mean = {np.mean(feature_maps_of_conv_layer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_conv_layer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}")
                                plt.imshow(feature_maps_of_conv_layer[0, kernel_id-1,:,:], cmap='gray')
                                st.pyplot(plt.gcf())
                                plt.close()
                                
                                if show_3d_plots:
                                    plot_filter_3d_plotly(filter_2d=kernel_dw_plot)
                                
                                if apply_norm_and_act and feature_maps_of_actlayer is not None and separate_norm_and_act == "No":   
                                    plt.figure(figsize=(10, 5))
                                    plt.subplot(1, 2, 1)
                                    plt.title("Normalization and Activation")
                                    plt.imshow(image_NormAct)
                                    plt.axis("off")
                                    plt.subplot(1, 2, 2)
                                    plt.title("Feature map after applying Normalization and Activation\n"
                                                f"Mean = {np.mean(feature_maps_of_actlayer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_actlayer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}")
                                    plt.imshow(feature_maps_of_actlayer[0, kernel_id-1,:,:], cmap='gray')
                                    st.pyplot(plt.gcf())
                                    plt.close()
                                    
                                elif apply_norm_and_act and feature_maps_of_actlayer is not None and separate_norm_and_act == "Yes":   
                                    ## Only normalization and its feature maps.
                                    plt.figure(figsize=(10, 5))
                                    plt.subplot(1, 2, 1)
                                    plt.title("Normalization after convolution")
                                    plt.imshow(image_Norm)
                                    plt.axis("off") 
                                    plt.subplot(1, 2, 2)
                                    plt.title("Feature map after applying Normalization\n"
                                                f"Mean = {np.mean(feature_maps_of_normlayer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_normlayer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}")
                                    plt.imshow(feature_maps_of_normlayer[0, kernel_id-1,:,:], cmap='gray')
                                    st.pyplot(plt.gcf())
                                    plt.close()
                                    
                                    ## Only activation and its feature maps.
                                    plt.figure(figsize=(10, 5))
                                    plt.subplot(1, 2, 1)
                                    plt.title("Activation after Normalization")
                                    plt.imshow(image_Act)
                                    plt.axis("off") 
                                    plt.subplot(1, 2, 2)
                                    plt.title("Feature map after applying Activation\n"
                                                f"Mean = {np.mean(feature_maps_of_actlayer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_actlayer[0, kernel_id-1,:,:].detach().cpu().numpy()):.3f}")
                                    plt.imshow(feature_maps_of_actlayer[0, kernel_id-1,:,:], cmap='gray')
                                    st.pyplot(plt.gcf())
                                    plt.close()
                                    
                                ##################################################
                                # Individual modificaiton - This is temporary
                                ##################################################
                                if required_weight_shape[-1] == 1:
                                    num_subkernels = network_weights[name_required_weight][kernel_id-1].shape[0]
                                    
                                    # Individual Threshold controls
                                    threshold_this_pw_kernel = st.checkbox(
                                        f'Apply threshold to kernel :green[#{kernel_id}] of layer {layer_mapping_dict[a_layer]}', 
                                        value=False,
                                        help="Thresholds the values of this kernel."
                                    ) 
                                    
                                    if threshold_this_pw_kernel:
                                        threhold_value_str_pw_kernel = st.text_input(f"Enter the threshold value for kernel :green[#{kernel_id}] of layer {layer_mapping_dict[a_layer]}",
                                                        "0.0")
                                        try:
                                            threhold_value_num_pw_kernel = float(threhold_value_str_pw_kernel)

                                        except:
                                            st.write(f"The current threshold value is invalid. Please enter a number and re-try, or disable ***threshold*** for this kernel.",)
                                            threhold_value_num_pw_kernel = 0.0
                                        
                                        original_pw_kernel = network_weights[name_required_weight][kernel_id-1]
                                        modified_pw_kernel = torch.where(torch.abs(original_pw_kernel) > threhold_value_num_pw_kernel, original_pw_kernel, 0)
                                        modified_pw_kernel_np = modified_pw_kernel.cpu().numpy().flatten()
                                        
                                        layer_name_corresponding_dw_conv = copy(a_layer) # Because I am being pedantic.
                                        layer_name_corresponding_dw_conv = layer_name_corresponding_dw_conv.replace("dws_conv.1", "dws_conv.0")
                                        feature_maps_of_corresponding_dw_conv = visualizer.get_feature_maps(input_tensor=input_image_tensor,
                                                                        layer_name=layer_name_corresponding_dw_conv) 
                                        
                                        name_required_bias = str(a_layer +".bias")
                                        feature_map_of_modified_pw_kernel = torch.nn.functional.conv2d(input=feature_maps_of_corresponding_dw_conv,
                                                                    weight=modified_pw_kernel.unsqueeze(0),
                                                                    bias=torch.tensor([network_weights[name_required_bias][kernel_id-1]]))                          
                                        
                                        plt.figure(figsize=(10, 5))
                                        plt.subplot(1, 2, 1)
                                        plt.plot(modified_pw_kernel_np, linewidth=1.0)
                                        xticks_step = math.ceil(kernel_pw_plot.shape[0]/16) # Why 16? Because 16 ticks fit in the plot. Deal with it!
                                        plt.xticks(np.arange(len(modified_pw_kernel_np),step=xticks_step), np.arange(1, len(modified_pw_kernel_np)+1, step=xticks_step))
                                        plt.grid()  
                                        plt.title(f"Modified Kernel after threshold = {threhold_value_num_pw_kernel} \n"
                                            f"Sum = {np.sum(modified_pw_kernel_np):.3f}")
                                        plt.subplot(1, 2, 2)
                                        plt.title("Feature map using modified Kernel\n"
                                                    f"Mean = {np.mean(feature_map_of_modified_pw_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_map_of_modified_pw_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}")
                                        plt.imshow(feature_map_of_modified_pw_kernel[0, 0,:,:], cmap='gray')
                                        
                                        st.pyplot(plt.gcf())
                                        plt.close()    
                                        
                                        if apply_norm_and_act and feature_map_of_modified_pw_kernel is not None:
                                            
                                            try:
                                                layer_norm_weight_after_modified_kernel = torch.tensor([network_weights[f"{layer_name_norm}.weight"][kernel_id-1]])
                                                layer_norm_bias_after_modified_kernel = torch.tensor([network_weights[f"{layer_name_norm}.bias"][kernel_id-1]])
                                                feature_maps_of_normlayer_after_modified_kernel = torch.nn.functional.instance_norm(input=feature_map_of_modified_pw_kernel,
                                                                                                                                weight=layer_norm_weight_after_modified_kernel,
                                                                                                                                bias=layer_norm_bias_after_modified_kernel)
                                            except:
                                                feature_maps_of_normlayer_after_modified_kernel = torch.nn.functional.instance_norm(input=feature_map_of_modified_pw_kernel)
                                                
                                            feature_maps_of_actlayer_after_modified_kernel = torch.nn.functional.leaky_relu(input=feature_maps_of_normlayer_after_modified_kernel)
                                            
                                            if separate_norm_and_act == "No":   
                                                
                                                plt.figure(figsize=(10, 5))
                                                plt.subplot(1, 2, 1)
                                                plt.title("Normalization and Activation")
                                                plt.imshow(image_NormAct)
                                                plt.axis("off")
                                                plt.subplot(1, 2, 2)
                                                plt.title("Feature map after applying Normalization and Activation\n"
                                                            f"Mean = {np.mean(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}")
                                                plt.imshow(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:], cmap='gray')
                                                st.pyplot(plt.gcf())
                                                plt.close()
                                                
                                            elif separate_norm_and_act == "Yes":   
                                                
                                                ## Only normalization and its feature maps.
                                                plt.figure(figsize=(10, 5))
                                                plt.subplot(1, 2, 1)
                                                plt.title("Normalization after convolution")
                                                plt.imshow(image_Norm)
                                                plt.axis("off") 
                                                plt.subplot(1, 2, 2)
                                                plt.title("Feature map after applying Normalization\n"
                                                            f"Mean = {np.mean(feature_maps_of_normlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_normlayer_after_modified_kernel[0, 0,:,:].item().detach().cpu().numpy()):.3f}")
                                                plt.imshow(feature_maps_of_normlayer_after_modified_kernel[0, 0,:,:], cmap='gray')
                                                st.pyplot(plt.gcf())
                                                plt.close()
                                                
                                                ## Only activation and its feature maps.
                                                plt.figure(figsize=(10, 5))
                                                plt.subplot(1, 2, 1)
                                                plt.title("Activation after Normalization")
                                                plt.imshow(image_Act)
                                                plt.axis("off") 
                                                plt.subplot(1, 2, 2)
                                                plt.title("Feature map after applying Activation\n"
                                                            f"Mean = {np.mean(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].item().detach().cpu().numpy()):.3f}")
                                                plt.imshow(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:], cmap='gray')
                                                st.pyplot(plt.gcf())
                                                plt.close()

                                    required_subkernel_ids = st.text_input(f"Enter the desired sub-kernel (position number) you want to retain from kernel :green[#{kernel_id}] of layer {layer_mapping_dict[a_layer]}. The un-selected sub-kernels will be zeroed out. If you want to display multiple sub-kernels, please separate ID values with commas (example: ***3*** or ***1,2,3*** or ***4,8,12*** or ***10-15*** or ***9,16-20*** or ***4-8,16-20*** or ***all*** or ***none***).", 
                                                            "none") 
                                    
                                    # To save some hassle.
                                    required_subkernel_ids_list = None
                                    try:
                                        if required_subkernel_ids == "none":
                                            pass
                                        else:
                                            if required_subkernel_ids.lower() == "all":
                                                required_subkernel_ids_list = [x for x in range(1,num_subkernels+1)]
                                                st.write(f"All kernels are selected, and their attributes will be displayed individually below.")  
                                            elif "-" in required_subkernel_ids and "," in required_subkernel_ids:
                                                required_subkernel_ids_list = []
                                                comma_split_list = required_subkernel_ids.split(",")
                                                for list_val in comma_split_list:
                                                    if "-" in list_val:
                                                        hypen_split_list = list_val.split("-")
                                                        if len(hypen_split_list) == 2:
                                                            required_subkernel_ids_list.extend([hypen_val for hypen_val in range(int(hypen_split_list[0]), int(hypen_split_list[1])+1)])
                                                        else:
                                                            st.write(f"The current value is invalid. Please enter valid kernel ID(s) in integer format and re-try.",)
                                                            required_subkernel_ids_list = None
                                                            break
                                                    else:
                                                        required_subkernel_ids_list.append(int(list_val))
                                                    
                                            elif "-" in required_subkernel_ids and "," not in required_subkernel_ids:
                                                required_subkernel_ids_list=[]
                                                hypen_split_list = required_subkernel_ids.split("-")
                                                if len(hypen_split_list) == 2:
                                                    required_subkernel_ids_list.extend([hypen_val for hypen_val in range(int(hypen_split_list[0]), int(hypen_split_list[1])+1)])
                                                else:
                                                    st.write(f"The current value is invalid. Please enter valid kernel ID(s) in integer format and re-try.",)
                                                    required_subkernel_ids_list = None
                                                    break
                                            else:
                                                required_subkernel_ids_list = [int(x) for x in required_subkernel_ids.split(",")]
                                                
                                            if any( x < 1 or x > num_subkernels for x in required_subkernel_ids_list):
                                                st.write(f"The current value is invalid. Please choose integer values between 1-{num_subkernels}.",)
                                                required_subkernel_ids_list = None
                                            else:  
                                                required_subkernel_ids_list = sorted(set(required_subkernel_ids_list))
                                                st.write(f"The selected kernel(s) are {required_subkernel_ids_list}, and only they will be retained.")        
                                    except:
                                        st.write(f"The current value is invalid. Please enter valid kernel ID(s) in integer format and re-try.",)
                                        required_subkernel_ids_list = None  
                                        
                                    if required_subkernel_ids_list is not None and input_image is None:
                                        st.write(f"Please upload the an input image to visualize the feature maps of the kernels.")
                                        
                                    elif required_subkernel_ids_list is not None and input_image is not None:
                                        original_pw_kernel = network_weights[name_required_weight][kernel_id-1]
                                        modified_pw_kernel = torch.zeros_like(original_pw_kernel)
                                        
                                        for subkernel in required_subkernel_ids_list:
                                            modified_pw_kernel[subkernel-1] = original_pw_kernel[subkernel-1]
                                        
                                        modified_pw_kernel_np = modified_pw_kernel.cpu().numpy().flatten()
                                        
                                        layer_name_corresponding_dw_conv = copy(a_layer) # Because I am being pedantic.
                                        layer_name_corresponding_dw_conv = layer_name_corresponding_dw_conv.replace("dws_conv.1", "dws_conv.0")
                                        feature_maps_of_corresponding_dw_conv = visualizer.get_feature_maps(input_tensor=input_image_tensor,
                                                                        layer_name=layer_name_corresponding_dw_conv) 
                                        
                                        name_required_bias = str(a_layer +".bias")
                                        feature_map_of_modified_pw_kernel = torch.nn.functional.conv2d(input=feature_maps_of_corresponding_dw_conv,
                                                                    weight=modified_pw_kernel.unsqueeze(0),
                                                                    bias=torch.tensor([network_weights[name_required_bias][kernel_id-1]]))                          
                                        
                                        plt.figure(figsize=(10, 5))
                                        plt.subplot(1, 2, 1)
                                        plt.plot(modified_pw_kernel_np, linewidth=1.0)
                                        xticks_step = math.ceil(kernel_pw_plot.shape[0]/16) # Why 16? Because 16 ticks fit in the plot. Deal with it!
                                        plt.xticks(np.arange(len(modified_pw_kernel_np),step=xticks_step), np.arange(1, len(modified_pw_kernel_np)+1, step=xticks_step))
                                        plt.grid()  
                                        plt.title(f"Modified Kernel after sub-kernel selection\n"
                                            f"Sum = {np.sum(modified_pw_kernel_np):.3f}")
                                        plt.subplot(1, 2, 2)
                                        plt.title("Feature map using selected sub-kernel(s)\n"
                                                    f"Mean = {np.mean(feature_map_of_modified_pw_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_map_of_modified_pw_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}")
                                        plt.imshow(feature_map_of_modified_pw_kernel[0, 0,:,:], cmap='gray')
                                        
                                        st.pyplot(plt.gcf())
                                        plt.close()    
                                        
                                        if apply_norm_and_act and feature_map_of_modified_pw_kernel is not None:
                                            
                                            try:
                                                layer_norm_weight_after_modified_kernel = torch.tensor([network_weights[f"{layer_name_norm}.weight"][kernel_id-1]])
                                                layer_norm_bias_after_modified_kernel = torch.tensor([network_weights[f"{layer_name_norm}.bias"][kernel_id-1]])
                                                feature_maps_of_normlayer_after_modified_kernel = torch.nn.functional.instance_norm(input=feature_map_of_modified_pw_kernel,
                                                                                                                                weight=layer_norm_weight_after_modified_kernel,
                                                                                                                                bias=layer_norm_bias_after_modified_kernel)
                                            except:
                                                feature_maps_of_normlayer_after_modified_kernel = torch.nn.functional.instance_norm(input=feature_map_of_modified_pw_kernel)
                                                
                                            feature_maps_of_actlayer_after_modified_kernel = torch.nn.functional.leaky_relu(input=feature_maps_of_normlayer_after_modified_kernel)
                                            
                                            if separate_norm_and_act == "No":   
                                                
                                                plt.figure(figsize=(10, 5))
                                                plt.subplot(1, 2, 1)
                                                plt.title("Normalization and Activation")
                                                plt.imshow(image_NormAct)
                                                plt.axis("off")
                                                plt.subplot(1, 2, 2)
                                                plt.title("Feature map after applying Normalization and Activation\n"
                                                            f"Mean = {np.mean(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}")
                                                plt.imshow(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:], cmap='gray')
                                                st.pyplot(plt.gcf())
                                                plt.close()
                                                
                                            elif separate_norm_and_act == "Yes":   
                                                
                                                ## Only normalization and its feature maps.
                                                plt.figure(figsize=(10, 5))
                                                plt.subplot(1, 2, 1)
                                                plt.title("Normalization after convolution")
                                                plt.imshow(image_Norm)
                                                plt.axis("off") 
                                                plt.subplot(1, 2, 2)
                                                plt.title("Feature map after applying Normalization\n"
                                                            f"Mean = {np.mean(feature_maps_of_normlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_normlayer_after_modified_kernel[0, 0,:,:].item().detach().cpu().numpy()):.3f}")
                                                plt.imshow(feature_maps_of_normlayer_after_modified_kernel[0, 0,:,:], cmap='gray')
                                                st.pyplot(plt.gcf())
                                                plt.close()
                                                
                                                ## Only activation and its feature maps.
                                                plt.figure(figsize=(10, 5))
                                                plt.subplot(1, 2, 1)
                                                plt.title("Activation after Normalization")
                                                plt.imshow(image_Act)
                                                plt.axis("off") 
                                                plt.subplot(1, 2, 2)
                                                plt.title("Feature map after applying Activation\n"
                                                            f"Mean = {np.mean(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].detach().cpu().numpy()):.3f}, Median = {np.median(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:].item().detach().cpu().numpy()):.3f}")
                                                plt.imshow(feature_maps_of_actlayer_after_modified_kernel[0, 0,:,:], cmap='gray')
                                                st.pyplot(plt.gcf())
                                                plt.close()
                            
                                        
                        st.divider()  # 👈 Draws a horizontal rule                

def main():
    MedicalImageExplorer.main()

if __name__ == '__main__':
    main()