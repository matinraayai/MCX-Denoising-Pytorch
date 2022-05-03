# A Framework for Denoising MonteCarlo Photon Transport Simulations Using Deep Learning
This repository contains the code for Journal of Biomedical Optics 
paper with the same name. This repo also contains additional code and results for denoising 2D simulations + additional 
test cases for both 2D and 3D.

# How to run
## Pre-requisites
All required Python packages can be installed via ```requirements.txt```:
```bash
pip install -r requirements.txt
```
Matlab also must be installed for generating datasets required for training and testing. MCXLab and other supporting
Matlab mex files are already included in this repository. There is no need to additionally clone the 
required files, unless you want a newer version of MCX/MCXLab. They 
are located in the ```matlab/``` folder.

## Dataset Generation
To generate both training and testing dataset, run Matlab function ```generate_data``` included in 
```data/generate/generate_data.m```. More info on the arguments can be found in the file itself.
## Training
To train, run ```train-lightning.py``` with the configs in the 2D and 3D folder. Refer to ```config.py``` for
more info on the config arguments.
## Inference
Run ```model-inference.py``` for model inference with configs in the 2D and 3D folder. Refer to ```config.py``` for
more info on the config arugments.

# Folder Structure
- ```configs/``` contains all the yaml configurations needed to run scripts
  - ```2D+3D/``` contains all configurations for 2D/3D fluence maps
    - ```analysis/``` contains all configurations for analysing inference results from all models
      - ```cross-section``` contains all configuration for analysing the middle cross-section of benchmarks B1-B3.                            
      - ```global-metrics``` contains all configurations for analysing the global metrics (MSE, PSNR, SSIM) for all 
                             benchmarks in the paper.
    - ```blind-denoising``` contains all configurations for training denoising models.
    - ```inference``` contains all configurations for performing inference on different datasets using different models.
    - ```profile``` (3D only) contains all configurations for profiling the denoising models on different dataset dimensions
                    presented in the paper.
    - ```visualization``` contains all configurations for visualizing the results acquired from denoising.
- ```data/```
  - ```augmentation/``` contains logic used for data augmentation during training.

## Benchmarking Denoisers

## Profiling Denoisers

## Questions

## Citation
If you use this work in your publication, please cite the following:
```bibtex

```