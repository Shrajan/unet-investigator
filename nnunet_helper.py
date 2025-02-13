import torch
import numpy as np
from pathlib import Path
from typing import Dict
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    
class nnUNetHelper:
    @staticmethod
    def predict_on_2d(network, input_array: np.ndarray) -> np.ndarray:
        network.eval()
        with torch.no_grad():  
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            input_image_tensor = torch.from_numpy(input_array.copy()).unsqueeze(0).unsqueeze(0).to(device)
            prediction_image_tensor = torch.argmax(network(input_image_tensor), dim=1)
            prediction_image = prediction_image_tensor[0,:,:].detach().cpu().numpy()
            del input_image_tensor, prediction_image_tensor
        return prediction_image
    
    @staticmethod
    def nnunet_load_weights(network: torch.nn.Module, trained_network_weights: Dict) -> torch.nn.Module:
        new_state_dict = {}
        for k, value in trained_network_weights.items():
            key = k
            if key not in network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
            
        network.load_state_dict(new_state_dict)    
        return network
    
    @staticmethod
    def nnunet_model_loading(savedCheckpointPath: Path):
        """
        Load the nnUNet model using the saved checkpoint. They are best suited for 2d and 3d_fullres
        configurations. I have not tried for DDP, 3d_lowres and cascades.

        Args:
            savedCheckpointPath (Path): Path to the saved checkpoint - nnUNet uses .pth file format.

        Returns:
            network: PyTorch model along with trained network weights using the saved plans and configuration information.
            network_weights: trained network weights.
            configuration: Whether it is '2d', '3d_fullres', etc.
            
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        model_checkpoint = torch.load(savedCheckpointPath, map_location=device)
        network_weights = model_checkpoint["network_weights"]
        model_configuration = model_checkpoint['init_args']['configuration']
        architecture_class_name = model_checkpoint['init_args']['plans']['configurations']['2d']['architecture']['network_class_name']
        arch_init_kwargs = model_checkpoint['init_args']['plans']['configurations']['2d']['architecture']['arch_kwargs']
        arch_init_kwargs_req_import = model_checkpoint['init_args']['plans']['configurations']['2d']['architecture']['_kw_requires_import']
        num_input_channels = len(model_checkpoint['init_args']['dataset_json']['channel_names'])
        num_output_channels = len(model_checkpoint['init_args']['dataset_json']['labels'])
        
        network = get_network_from_plans(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                allow_init=True,
                deep_supervision=True).to(device)
        
        return network, network_weights, model_configuration, arch_init_kwargs

if __name__ == '__main__':
    nnUNetHelper.nnunet_model_loading(savedCheckpointPath="checkpoint_latest.pth")