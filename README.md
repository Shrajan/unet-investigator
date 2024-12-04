# What happens under the hood of a U-Net.
This repository helps in understanding the inner working of the U-Nets with the aid of interactive visualization. The application runs locally on the device, and is opened on the browser.

## U-Net Architectures
* To best understand the kernels and their features, it is recommended to train a `2D` model with kernel size of at least `5`. 
* The checkpoint files provided in this repository were trained on the `AMOS-2022` dataset with a kernel size of `7` using the `nnU-Net v2` framework.
* Furthermore, unlike the default U-Net architecture of the nnU-Net framework, our implementation uses `Depth-wise separable` convolutions (except the first layer of Encoder 1).

## System Requirements

### Operating System
The application has been tested on Ubuntu (18.04, 20.04, 22.04), Windows and MacOS (including M1).
  
### Hardware 
It is recommended to have sufficient RAM (minimum 8 GB), to ensure a smooth operation. Having a GPU would be beneficial, as it would take some load off the CPU.

## Installation
We used **Python**, along with supporting libraries (*Streamlit*), to create the application. We recommend that you install all the necessary packages in a virtual environment, such as **pip** or **anaconda**. Please use a recent version of Python (>3.10).

1) Download/clone the repository, and navigate to the project directory: `cd unet-investigator`.
2) Make sure to have the latest pip version. Run the this command to retrieve it: `pip install --upgrade pip`
3) The models were trained using the *nnU-Net* framework, that uses **PyTorch**. Kindly install [PyTorch](https://pytorch.org/get-started/locally/) as suggested on their website (conda/pip), and make sure you install the latest version with support for your hardware (Cuda, MPS, CPU).
4) Install **nnU-Net**: `pip install nnunetv2`
5) Additional U-Net variants (Attention U-Net, IB-U-Nets, etc.), and with *Depth-wise separable convolutions* have been added, and these can be availed by re-installing my version of the dynamic-network-architectures repository:
    ```
    git clone https://github.com/Shrajan/dynamic-network-architectures.git
    cd dynamic-network-architectures
    pip install -e .
    cd ../
    ```
6) Finally, install **Streamlit**: `pip install streamlit`
7) For improved performance, we recommend to install **watchdog**: `pip install watchdog`

## Usage
To run the application, simply execute: `streamlit run unet_vis.py`

### Numpy errors
If you get *numpy* problems with PyTorch, please change the numpy version. Something like this.
`pip install --force-reinstall -v "numpy==1.26.3"`


