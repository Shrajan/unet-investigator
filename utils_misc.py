from typing import List, Dict, Tuple

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