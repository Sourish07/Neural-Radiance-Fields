# Neural-Radiance-Fields

This repository contains the accompanying code for my [YouTube video](https://youtu.be/XANrj4-DNSc) about Neural Radiance Fields! We implement the tiny NeRF variant from the paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Ben Mildenhall et al. The code is written in PyTorch and is heavily inspired by the official NeRF implementation by the authors.

The v1 branch contains a basic NeRF implementation, resulting in a blob of a Cybtertruck.

The v2 branch adds the positional encoding feature, resulting in a more detailed Cybertruck.

The v3 branch adds some checkpointing code, animation code, along with the code used in Blender to generate the dataset.

## Running the code locally

1. Clone the repository
2. Create a virtual environment and install the dependencies. Use `requirements.txt` if you're using an NVIDIA GPU and `requirements-cpu.txt` if you're using a CPU.
```bash
conda create --name nerf python=3.11
conda activate nerf
pip install -r requirements.txt
```

3. Run the code in main.ipynb