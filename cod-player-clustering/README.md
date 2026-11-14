# Call of Duty Player Clustering

Unsupervised segmentation of Call of Duty players from their match statistics, comparing K-Means, agglomerative (Ward) clustering and DBSCAN. The project also includes a from-scratch DBSCAN implementation, benchmarked against scikit-learn.

## Motivation

Grouping players by how they actually play is useful for matchmaking, balancing and understanding a player base. Density-based methods such as DBSCAN also label points that do not belong to any dense region as noise. In player data, those noise points are players whose stat profile is unusual compared to everyone else. That makes them a reasonable starting point for anomaly analysis, for example a short playtime combined with exceptional performance.

This is an exploratory clustering project, not a cheat detector. An outlier here only means "statistically unusual". It does not mean "cheating".

## Dataset

- [Call of Duty Players](https://www.kaggle.com/datasets/aishahakami/call-of-duty-players) (Kaggle, uploaded by `aishahakami`), stored in `data/cod.csv`
- 1,558 players × 19 columns: wins, kills, kdRatio, killstreak, level, xp, prestige, hits, shots, misses, headshots, timePlayed, gamesPlayed, scorePerMinute, deaths, and more
- The DBSCAN-from-scratch notebook uses the Iris dataset that ships with scikit-learn, so it needs no data file.

## Approach

### `cod_player_clustering.ipynb`

1. **EDA:** data types, missing values, duplicates, summary statistics, histograms and correlation matrices.
2. **Cleaning:** removed 682 players with zero values in key activity columns (hits, shots, misses, headshots, scorePerMinute, wins, gamesPlayed), leaving 876 players. An IQR outlier filter is documented but deliberately not applied, so that DBSCAN can still find the outliers.
3. **Feature engineering:** replaced highly correlated raw counts with ratio features: `accuracy`, `miss_rate`, `headshot_rate`, `score_rate`, `winrate` and `trueLevel` (XP per level). These are combined with `kdRatio`, `killstreak`, `timePlayed` and `gamesPlayed`, giving 10 features.
4. **Clustering experiments**, each visualized with a 2D PCA projection and profiled by per-cluster feature means:
   - K-Means + StandardScaler (k chosen with the elbow method and silhouette score)
   - K-Means + log1p transform + StandardScaler
   - DBSCAN + StandardScaler (Manhattan distance, eps chosen from a k-distance plot)
   - Agglomerative clustering (Ward) + MinMaxScaler (k chosen from a dendrogram)

### `dbscan_from_scratch.ipynb`

- Covers the DBSCAN theory: ε-neighborhoods, core, border and noise points, and density reachability and connectivity.
- Implements `DBSCANFast`, which precomputes ε-neighborhoods with `NearestNeighbors.radius_neighbors` and expands clusters with a BFS queue. The original naive version was O(n²) and is not included.
- Compares it with `sklearn.cluster.DBSCAN` on standardized Iris (eps = 0.9, minPts = 3): cluster count, noise count, silhouette score and average runtime over 100 runs.

## Results

**Player clustering** (876 players):

| Experiment | Clusters | Cluster sizes | Silhouette |
|---|---|---|---|
| K-Means + StandardScaler | 3 | 464 / 241 / 171 | 0.2698 |
| K-Means + log + StandardScaler | 3 | 532 / 276 / 68 | 0.3744 |
| DBSCAN (eps = 5.5, min_samples = 10, Manhattan) | 1 + noise | 860 / 16 noise | not computed |
| Agglomerative (Ward) + MinMaxScaler | 4 | 334 / 319 / 119 / 104 | not computed |

The two K-Means silhouette scores were computed in different feature spaces, so they are not directly comparable.

- **K-Means (standard scaling)** finds three groups: newcomers (lowest K/D, level and playtime), average players, and top players (highest K/D, level, playtime and killstreak).
- **K-Means (log transform)** separates a small cluster of 68 players with high playtime but few matches, an unusually high headshot rate and a very high winrate. The notebook flags this group as inconsistent, possibly due to incomplete data or suspicious behaviour.
- **DBSCAN** finds one dense core of typical players and 16 noise points. The noise points have higher average K/D, killstreak, headshot rate and winrate than the core group, so these are the players that would deserve a closer look.
- **Ward clustering** produces four skill tiers: newcomers or weakest players, below-average, average, and top players.

**Custom DBSCAN vs scikit-learn** (Iris, 150 samples):

| Implementation | Clusters | Noise points | Silhouette | Avg. runtime (100 runs) |
|---|---|---|---|---|
| Custom `DBSCANFast` | 2 | 4 | 0.52 | 0.0023 s |
| scikit-learn `DBSCAN` | 2 | 4 | 0.52 | 0.0016 s |

Both implementations give the same cluster count, noise count and silhouette score. The custom version is roughly 1.4× slower than scikit-learn (timings are machine-dependent).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook
```

Run the notebooks from the project folder, so that the relative path `data/cod.csv` resolves.

## File structure

```
cod-player-clustering/
├── cod_player_clustering.ipynb   # EDA, feature engineering, K-Means / DBSCAN / Ward experiments
├── dbscan_from_scratch.ipynb     # custom DBSCAN implementation + benchmark vs scikit-learn
├── data/
│   └── cod.csv                   # Kaggle Call of Duty players dataset
├── requirements.txt
└── README.md
```
