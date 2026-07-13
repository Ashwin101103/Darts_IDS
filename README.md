# DARTS-IDS: Neural Architecture Search for Network Intrusion Detection

This repository contains the full experimental pipeline for **DARTS-IDS**, a Differentiable Architecture Search (DARTS) based Intrusion Detection System evaluated on the **NF-ToN-IoT-v2** network flow dataset. The pipeline covers architecture search, final model training/evaluation, baseline comparisons, class-aware post-training quantization (CAQ), and ONNX export for edge deployment.

## Overview

The notebook implements an end-to-end research pipeline:

1. **Data loading & preprocessing** - loads NF-ToN-IoT-v2, encodes labels, scales features, and performs a stratified train/search/test split.
2. **DARTS architecture search** - a supernet made of 4 mixed-operation cells (`linear_relu`, `bottleneck`, `residual`, `se_block`, `gated_linear`) is trained with bi-level optimization (weights vs. architecture parameters/alphas) to discover the best operation per cell.
3. **Final model training** - the discovered architecture (`FinalDARTS`) is retrained from scratch on the combined train+search split and evaluated on the held-out test set.
4. **Evaluation & visualization** - confusion matrix, ROC curves, precision-recall curves, and calibration curves for the final model.
5. **Baseline comparison** - DARTS-IDS is benchmarked against an MLP (with/without class weights), XGBoost, LightGBM, Random Forest, and Logistic Regression, including a Gaussian-noise robustness sweep.
6. **Class-aware quantization (CAQ)** - a greedy, class-aware search that restores selected FP32 modules on top of an INT8-quantized model to recover accuracy on hard classes while minimizing size.
7. **ONNX static quantization** - exports the model to ONNX and searches over node-exclusion combinations to find the best accuracy/size trade-off for static INT8 quantization, targeting microcontroller-class deployment (e.g., ESP32).
8. **Paper table generation** - produces the final CSV tables (main comparison, calibration ablation, incremental gains, compression/performance trade-off, summary) used in the paper.

## Repository Structure

```
.
├── darts-ids.ipynb        # Main notebook (all pipeline blocks)
├── README.md
└── paper_results/          # Generated at runtime — see below
```

All intermediate artifacts (trained weights, JSON metrics, ONNX models, CSV tables) are written to a `paper_results/` directory that is created automatically when the notebook runs. This directory is not tracked in git (see `.gitignore` suggestion below).

## Dataset

- **Name:** NF-ToN-IoT-v2
- **Format:** CSV, expected at a path such as `/kaggle/input/datasets/<user>/nf-ton/NF-ToN-IoT-v2.csv` (update `CSV_PATH` in the first cell to match your environment).
- The dataset is **not included** in this repository. Download it separately and update the path before running.

## Requirements

- Python 3.9+
- PyTorch (with CUDA optional but recommended for the search/training blocks)
- numpy, pandas, scikit-learn, matplotlib, seaborn
- xgboost, lightgbm
- onnx, onnxruntime, onnxruntime-tools (for the ONNX quantization blocks)

Install with:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm onnx onnxruntime onnxruntime-tools
```

## Usage

1. Update `CSV_PATH` in the data-loading cell to point to your local copy of NF-ToN-IoT-v2.
2. Run the notebook top to bottom. Note the notebook was developed iteratively on Kaggle, so a few cells are alternate/duplicate versions of the same block (e.g., there are two DARTS search cells with slightly different hyperparameters, and two checkpoint-loading cells for different loading strategies) — kept here for transparency into the experimental process. Run the variant that matches the checkpoint/config you intend to reproduce.
3. Outputs (model checkpoints, metrics, tables, ONNX artifacts) are saved under `paper_results/`.

## Notes

- Some cells contain `!pip install ...` lines for the Kaggle environment (ONNX/ONNX Runtime); install these locally if not already present instead of relying on the shell command.
- The notebook assumes CUDA GPU availability for reasonable runtimes on the search and baseline-comparison blocks, but will fall back to CPU automatically.
- Numeric results embedded in later cells (e.g., baseline comparison table, quantization tables) reflect specific runs from the author's experiments and are provided for reference/reproducibility checks — re-running the notebook end-to-end will regenerate these from scratch.

## Citation

If you use this code in your research, please cite the corresponding paper (citation details to be added upon publication).

## License

Add a license of your choice (e.g., MIT) before making this repository public.
