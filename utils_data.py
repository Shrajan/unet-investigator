from typing import List, Optional, Tuple, Union, Dict
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def preprocess_ct_slice(image):
    return image

def normalize_image(image):
    return image

def apply_windowing(image):
    return image

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