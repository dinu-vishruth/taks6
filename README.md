# KNN Classification

## Overview
Implemented K-Nearest Neighbors classifier on Iris dataset. Preprocessing includes standard scaling. Evaluations: validation accuracy, confusion matrix, classification report. Visualized k vs accuracy and decision boundaries (PCA-2D).

## Files
- src/knn_task.py : main script (run with `python src/knn_task.py`)
- results/ : contains plots and saved model
- requirements.txt

## How to run
1. Create venv & install dependencies: `pip install -r requirements.txt`
2. Run: `python src/knn_task.py`
3. Check `results/` for accuracy/decision boundary images and saved model.

## Notes
- Experiment with different k values in the script.
- To use another dataset: replace `load_data()` with reading from CSV and follow same pipeline.
