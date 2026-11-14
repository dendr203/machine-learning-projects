# Machine Learning Projects

A collection of interesting machine learning projects I built during my Computer Science (BSc) and Cybersecurity (MSc) studies at VSB – Technical University of Ostrava. They cover classical ML on game data, deep learning with PyTorch, unsupervised player profiling, and search-based game AI.

| Project | Topic | Tech | Headline result |
|---|---|---|---|
| [**League of Legends Win Prediction**](lol-win-prediction/) | Predict the match winner from the first 15 minutes of ~24k ranked SoloQ games | pandas, scikit-learn, XGBoost, Plotly | **75.2 %** test accuracy, **AUC 0.83** (Logistic Regression ≈ tuned RF/XGBoost) |
| [**FER-2013 Emotion Recognition**](fer2013-emotion-recognition/) | 7-class facial emotion classification with a heavily imbalanced dataset | PyTorch, torchvision, Optuna, ONNX Runtime, OpenCV | ResNet18 transfer learning: **65.7 %** accuracy, **0.649 Macro-F1**; custom CNN exported to ONNX |
| [**Call of Duty Player Clustering**](cod-player-clustering/) | Segment players by behaviour; density-based detection of statistical outliers. Includes DBSCAN written from scratch | scikit-learn, SciPy, seaborn | 3 player archetypes (silhouette 0.37); DBSCAN isolates 16 anomalous high-performers |
| [**2048 AI Agents**](2048-ai/) | Greedy → weighted heuristic → expectimax agents, benchmarked over 30 games each | Python, NumPy | Improved expectimax: avg. score **13,022** and avg. max tile **863**, ~4× the greedy baseline's score |

## Highlights

- **Game data and competitive integrity.** Two projects work with real player data (League of Legends, Call of Duty). The clustering project uses density-based methods to flag players whose stats are far outside the normal population, which is the first step of any statistical cheat-detection pipeline.
- **End-to-end deep learning.** The FER project goes from EDA and class-imbalance handling (weighted sampling, class weights, Macro-F1) through Optuna hyperparameter search and transfer learning to **ONNX export and a standalone inference class**.
- **Careful evaluation.** Stratified splits, cross-validated hyperparameter search, confusion matrices, ROC/AUC and per-class metrics. Data-quality problems found in the source dataset (e.g. swapped columns, impossible values) are documented and fixed.
- **Algorithms from scratch.** A custom DBSCAN implementation validated against scikit-learn, plus a 2048 engine with expectimax search.

## Repository structure

```
machine-learning-projects/
├── lol-win-prediction/          # Notebook, dataset, README
├── fer2013-emotion-recognition/ # Notebook, plots, exported ONNX model, README
├── cod-player-clustering/       # Clustering notebook, DBSCAN-from-scratch notebook, dataset
└── 2048-ai/                     # Agents notebook, base game, full experiment report
```

Each project folder has its own README with the dataset source, methodology, results and instructions for running it.

## Running the projects

```bash
git clone https://github.com/dendr203/machine-learning-projects.git
cd machine-learning-projects/<project>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook
```

The FER-2013 dataset is not included because of its size. See [its README](fer2013-emotion-recognition/README.md) for download instructions.

## Notes

These projects started as university coursework. I translated them from Czech into English for this repository, cleaned up the code and re-executed the notebooks where possible, so the saved outputs match the code. The FER-2013 models could not be retrained for this release, so some of its embedded plots still have Czech labels. AI assistance used in the 2048 project is disclosed in its [report](2048-ai/REPORT.md).
