# pth-root-coherence-factor-NDT

GPU-accelerated MATLAB implementation of **p-th-root coherence-factor weighted delay-and-sum (pCFwDAS)** beamforming for  non-destructive testing (NDT), developed at **MUSE Lab, IIT Gandhinagar**.

---
## Key Files

- `pthcoherencefactor.m` — p-th-root CF (CPU)
- `pthcoherenceNDT.m` — CPU wrapper
- `pthcoherenceNDT_GPU.cu` — CUDA kernel source
- `pthcoherenceNDT_GPU.mexw64` — compiled GPU binary (for Windows)
- `pcfwDASGPU.m` — main pCFwDAS GPU beamformer
- `pcfwdasCPU.m` — Example Image reconstruction script for CPU
- `pcfwdasGPU.m` — Example Image Reconstruction Script for GPU
 - `b_mode_NDT_FMC_10V2025-10-22-p10` — sample dataset


## Requirements
- MATLAB R2021a or later
- Parallel Computing Toolbox (recommended)
- CUDA-capable GPU (for GPU execution)
- CUDA Toolkit installed (tested on CUDA 11.x / 12.x)

---

## Installation
Clone the repository and enter it:
```bash
git clone https://github.com/muselab-iitgn/pth-root-coherence-factor-NDT.git
cd pth-root-coherence-factor-NDT
```

## Compilation
Compile the CUDA kernel with : 
```bash
mexcuda pthcoherenceNDT_GPU.cu
```

## Verification of GPU Support
Check GPU compatibility with: 
```bash
% Check GPU availability
gpuDevice
```

## Usage
Ensure your dataset is in the same path. Once the imaging parameters are set, the beamforming functions can be called like : 
```bash
    [I] = pthcoherenceNDT_GPU(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time,fs,x_range, z_range, Single_element_loc,p); % For GPU
    [I] = pthcoherenceNDT(rf_Data, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time,fs,x_range, z_range, Single_element_loc,p); % For CPU


```
Repeat over chosen elements and sum to form full image.
`pcfwdasGPU.m` can be used as an example script to generate beamformed images.




