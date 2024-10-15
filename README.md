# NumpyGPT

NumpyGPT is a PyTorch-like implementation of a GPT (Generative Pre-trained Transformer) model using NumPy and CuPy for GPU acceleration. This project aims to provide a lightweight and educational version of the GPT architecture, suitable for learning and experimentation.
Inspired by NanoGPT.
## Features

- Implements core GPT architecture components
- Uses NumPy for CPU operations and CuPy for GPU acceleration
- Includes modules for embedding, self attention, dropout, and layer normalization
- Supports configurable model parameters (e.g., vocabulary size, embedding size, number of layers)
- Provides training loop with learning rate scheduling and gradient accumulation
- Includes text generation capabilities

## Requirements

- NumPy
- CuPy (for GPU acceleration)

## Usage

- `python3 train.py`: run for start train
- `python3 sample.py`: run for start inference model. Data taken from last checkpoint by default


## Project Structure

The project consists of several Python files that implement different components of the GPT model:

- `model_gpu.py`: Contains the main GPT model implementation
- `train.py`: Contains the train code
- `utils.py`: Some neccesary functions for import and export weights
- `sample.py`: Inference code

## Model Architecture

NumpyGPT implements a transformer-based language model with the following key components:

- Token and position embeddings
- Multi-head self-attention mechanism
- Feed-forward neural networks
- Layer normalization
- Dropout for regularization

## Training

The model supports training with:

- Customizable learning rate scheduling
- Gradient accumulation for effective batch size control
- Checkpointing for saving and resuming training
