# DGKL: Deep Graph Kernel Learning for Uncertainty Quantification in Catalysis

A Python-based project focused on uncertainty quantification for catalytic data analysis.

## Overview

Implements the code for the paper "Deep Graph Kernel Learning for Material & Atomic Level Uncertainty Quantification in Adsorption Energy Prediction".

## Project Structure

- `cat_uncertainty/`: Main package containing the core implementation
- `experiments/`: Directory containing experimental scripts. Results are not included due to size limit.

## Dependencies

The project uses Poetry for dependency management. Key dependencies include:
- Python 3.10+
- PyTorch with CUDA support
- PyTorch Geometric
- GPyTorch
- ASE (Atomic Simulation Environment)
- PyTorch Lightning
- Weights & Biases for experiment tracking

## Installation

1. Ensure you have Poetry installed
2. Clone the repository
3. Install dependencies:
   ```bash
   poetry install
   ```

## License

MIT License

Last updated: March 17, 2025