# Note from the author Shrajan.
# If you reading this, then I am sorry for what you are about to see. 
# This piece of code is absolutely basic, crudely written, unoptimized, and straight up madness.
# It is possible that some things may not work for you or it might crash.
# So I urge you to stop here, and go no further. STOP {whatever you like to be called}!!!
# Even after reading the above warnings, if you proceed, then I am not to be blamed.
# DO THIS AT YOUR OWN PERIL. 

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import pandas as pd
from typing import List, Optional, Tuple, Union

# Assuming you have these utility functions in separate files
from nnunet_helper import nnunet_model_loading
from preprocessing_utils import preprocess_ct_slice

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


class AdvancedCNNVisualizer:
    def __init__(self, model):
        self.model = model
    
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
                # Apply thresholding
                thresholded_output = torch.where(
                    output > threshold, 
                    output, 
                    torch.zeros_like(output)
                )
                feature_maps.append(thresholded_output.detach())
            
            layer = dict(self.model.named_modules())[layer_name]
            handle = layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                self.model(input_tensor)
            
            handle.remove()
            return feature_maps[0]
        except Exception as e:
            st.error(f"Error extracting feature maps: {e}")
            return None

    def visualize_feature_maps(self, 
                                input_image: torch.Tensor, 
                                layer_name: str, 
                                max_maps: int = 16, 
                                cmap: str = 'viridis', 
                                threshold: float = 0.0) -> Optional[plt.Figure]:
        """
        Advanced feature map visualization with multiple options
        
        Args:
            input_image (torch.Tensor): Input image tensor
            layer_name (str): Layer to extract features from
            max_maps (int): Maximum number of feature maps to display
            cmap (str): Colormap for visualization
            threshold (float): Threshold to apply to feature maps
        
        Returns:
            Matplotlib figure with feature maps
        """
        feature_maps = self.get_feature_maps(input_image, layer_name, threshold)
        
        if feature_maps is None:
            return None
        
        # Select first batch
        feature_maps = feature_maps[0]
        
        # Limit number of maps
        n_maps = min(max_maps, feature_maps.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_maps)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.ravel() if grid_size > 1 else [axes]
        
        for i in range(n_maps):
            feature_map = feature_maps[i].cpu().numpy()
            
            # Additional feature map analysis
            map_stats = {
                'Mean': np.mean(feature_map),
                'Max': np.max(feature_map),
                'Min': np.min(feature_map),
                'Std': np.std(feature_map)
            }
            
            im = axes[i].imshow(feature_map, cmap=cmap)
            axes[i].axis('off')
            
            # Annotate with statistics
            axes[i].set_title(
                f'Map {i+1}\n'
                f'Mean: {map_stats["Mean"]:.2f}\n'
                f'Max: {map_stats["Max"]:.2f}', 
                fontsize=8
            )
        
        # Hide unused subplots
        for j in range(n_maps, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        return fig

    def advanced_kernel_analysis_1d(self, 
                                  layer_name: str, 
                                  cmap: str = 'viridis', 
                                  normalize: bool = True,
                                  num_columns: int = 1,
                                  figsize: Tuple =(12, 12),
                                  fontsize: int =8,
                                  apply_threshold: bool = False,
                                  threshold_value: float = 0.0 ) -> Tuple[plt.Figure, dict]:
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
        n_columns = num_columns
        grid_size = int(np.ceil(n_kernels/n_columns))
        
        fig, axes = plt.subplots(grid_size, n_columns, figsize=figsize)
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
                'sum': np.sum(kernel),
                'mean': np.mean(kernel),
                'std': np.std(kernel),
                'max': np.max(kernel),
                'min': np.min(kernel)
            }
            kernel_stats['kernel_details'].append(kernel_info)
            
            im = axes[i].plot(kernel, linewidth=1.0)
            axes[i].set_xticks(np.arange(len(kernel)), np.arange(1, len(kernel)+1))
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
                    'sum': np.sum(kernel),
                    'mean': np.mean(kernel),
                    'std': np.std(kernel),
                    'max': np.max(kernel),
                    'min': np.min(kernel)
                }
                kernel_stats['kernel_details'].append(kernel_info)
                
                im = axes[i].imshow(kernel, cmap=cmap)
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
        st.sidebar.header('Model Configuration')
        uploaded_model = st.sidebar.file_uploader(
            "Upload PyTorch Model", 
            type=['pth', 'pt'],
            help="Upload a pre-trained U-Net model in PyTorch format"
        )
        st.sidebar.divider()  # 👈 Draws a horizontal rule
        
        if uploaded_model is None:
            st.markdown("""
                ### 🚀 To get started
                - Please upload a pre-trained U-Net model under the `Model Configuration` tab in the sidebar.
                - At this moment, we are supporting U-Net models created using the ***nnUNet V2*** framework.
                - It is preferred to use a ***2D*** configuration, as 3D is difficult to visualize and understand. 
                - We are using this to understand the inner workings of ***Depth-wise Separable*** convolutions.
                """)
            st.divider()  # 👈 Draws a horizontal rule

        elif uploaded_model is not None:
            # Model loading
            try:
                network, network_weights, network_configuration = nnunet_model_loading(savedCheckpointPath=uploaded_model)
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
                
                # TODO: Message - model loading successful.
                # TODO: Architecture - image of the trained model - Hidden layer or PyTorchViz
                
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
                        plt.imshow(original_slice, cmap='gray')
                        plt.subplot(1, 2, 2)
                        plt.title("Preprocessed Image")
                        plt.imshow(input_image, cmap='gray')
                        st.pyplot(plt.gcf())
                        st.divider()  # 👈 Draws a horizontal rule
                
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
                available_layers = [
                    name for name, module in network.named_modules() 
                    if isinstance(module, torch.nn.Conv2d)
                ]
                
                for idx, a_layer in enumerate(available_layers):
                    
                    # Kernel analysis controls
                    layer_selection_button = st.sidebar.checkbox(
                        f'Click to choose feature maps of {a_layer}', 
                        value=False,
                    )

                    if layer_selection_button:
                        
                        # Name of selected layer
                        #st.subheader(f'Displaying kernels of layer: ***{a_layer}***')
                        name_required_weight = str(a_layer +".weight")
                        required_weight_shape = network_weights[name_required_weight].shape
                        
                        st.markdown(f"""
                        #### Displaying kernels of layer: {a_layer}
                        This layer has ***{required_weight_shape[0]}*** kernels. 
                        You can choose to display ***all***, and/or enter below your desired kernel ID.
                        """)
                        
                        # Display all kernels controls
                        display_all_kernels = st.checkbox(
                            f'Display ***all*** kernels of layer: {a_layer}', 
                            value=True,
                            #help="Scales kernel values to a consistent range for better comparison"
                        )
                        
                        if display_all_kernels:
                            st.write("##### Plots with Original Values/Distribution")
                            
                            # Some variables to prevent chaos.
                            kernel_fig = None
                            threshold_all_plots = False
                            
                            with st.expander(f"Control panel of layer: {a_layer}"):
                                if required_weight_shape[-1] == 1:
                                    # This does not matter.
                                    cmap_kernels = 'viridis'
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # For setting the height and width of the plots.
                                        layer_plot_height = st.slider(f"Plot Height for layer: {a_layer}", 1, 100, 60)
                                        layer_plot_width = st.slider(f"Plot Width for layer: {a_layer}", 1, 50, 10)
                                        layer_figsize = (layer_plot_width, layer_plot_height)
                                        
                                        # Threshold controls
                                        threshold_all_plots = st.checkbox(
                                            f'Apply threshold and create new plot for layer: {a_layer}', 
                                            value=False,
                                            help="Thresholds all the values of the layer"
                                        )
                                        
                                    with col2:
                                        # Plot columns
                                        layer_plot_columns = st.slider(f"Number of columns for layer: {a_layer}", 1, required_weight_shape[0], 1)
                                        
                                        # Plot font size
                                        layer_plot_font = st.slider(f"Font-Size for layer: {a_layer}", 1, 25, 12)

                                        # Kernel analysis controls
                                        normalize_kernels = st.checkbox(
                                            f'Normalize kernel values for layer: {a_layer}', 
                                            value=False,
                                            help="Scales kernel values to a consistent range for better comparison"
                                        )
                                        
                                    # Generate kernel visualization
                                    kernel_fig, kernel_stats = visualizer.advanced_kernel_analysis_1d(
                                        layer_name=a_layer, 
                                        normalize=normalize_kernels,
                                        num_columns=layer_plot_columns,
                                        figsize=layer_figsize,
                                        fontsize=layer_plot_font
                                    )
                                     
                                    
                                else:
                                    layer_figsize = (12,12)
                                    col1, col2 = st.columns(2)
                            
                                    with col1:
                                        # Common colormap selection
                                        cmap_kernels = st.selectbox(
                                            f'Colormap for all kernels of layer: {a_layer}', 
                                            ['viridis', 'gray', 'plasma', 'inferno', 'magma', 'cividis', 'jet']
                                        )
                                    
                                    with col2:
                                        # Plot font size
                                        layer_plot_font = st.slider(f"Font-Size for layer: {a_layer}", 1, 25, 12)

                                    # Kernel analysis controls
                                    normalize_kernels = st.checkbox(
                                        f'Normalize kernel values for layer: {a_layer}', 
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
                            
                                if threshold_all_plots:
                                    st.write(" . "*98) # Found this value using trial and error for my machine - it is not dynamic because for some reason streamlit does not give page width. And I was lazy to search further.
                                    st.write("##### Plots with Threshold Values")
                                    threhold_value_str = st.text_input(f"Enter the threshold value for: {a_layer}",
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
                                        
                                    except:
                                        st.write(f"The current threshold value is invalid. Please enter a number and re-try, or disable ***threshold*** in the control panel.",)
                                            
                        
                        st.write(" . "*98) # Found this value using trial and error for my machine - it is not dynamic because for some reason streamlit does not give page width. And I was lazy to search further.
                        required_kernel_ids = st.text_input(f"Enter the required kernel ID for: {a_layer}. If you want to display multiple kernels, please separate ID values with commas (example: 3 or 1,2,4 or 4,8,12).", 
                                                            "none")
                        #if len(required_kernel_ids) 
                        st.write(f"The current selection is", required_kernel_ids)
                        st.write("The len of current selection is", len(required_kernel_ids))
                        
                        
                        
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
        
        st.divider()  # 👈 Draws a horizontal rule
        st.title("That's what happens inside a U-Net!")                

def main():
    MedicalImageExplorer.main()

if __name__ == '__main__':
    main()