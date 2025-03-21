# DGKL: Deep Graph Kernel Learning for Uncertainty Quantification in Catalysis

[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-Latest-blue.svg)](https://chemrxiv.org/engage/chemrxiv/article-details/67dc2d6381d2151a022b2aad)

DGKL is a deep graph kernel learning framework for uncertainty quantification in catalysis, specifically designed for accurate prediction and uncertainty estimation of adsorption energies in heterogeneous catalysis.

## 🚀 Environment Setup
- System requirements: This package requires a Linux computer with GPU. The codes have been tested on NVIDIA GPUs (A100 on CAC cluster, V100 on Expanse cluster, and RTX 4060 on local machine).
- We use `poetry` for dependency management. Key dependencies include:
  - Python 3.10+
  - PyTorch with CUDA support
  - PyTorch Geometric
  - GPyTorch
  - ASE (Atomic Simulation Environment)
  - PyTorch Lightning
  - Weights & Biases (optional)

### Installation
1. Ensure you have Poetry installed
2. Clone the repository
3. Install dependencies:
   ```bash
   poetry install
   ```

## 📌 Project Structure
```
cat_uncertainty/    # Main package containing core implementation
experiments/       # Experimental scripts and configurations
Paper/             # Paper pdf file
```

## 🔥 Model Training
Details about model training and experiments can be found in our paper. Results and lmdb files are not included in the repository due to size limitations.

## ⭐ Acknowledgements
This work builds upon several excellent open-source projects:
- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- GPyTorch
- ASE (Atomic Simulation Environment)
- fairchem

## 📝 Citation
If you find our work useful, please consider citing it:
```bibtex
@article{mamun2025deep,
  title={Deep Graph Kernel Learning for Material and Atomic Level Uncertainty Quantification in Adsorption Energy Prediction},
  author={Mamun, O and Yang, C and Yue, S},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-pfng2-v2},
  note={Preprint}
}
```

## 📫 Contact
If you have any questions, please contact:

Osman Mamun (mamun.che06@gmail.com)

## License
MIT License