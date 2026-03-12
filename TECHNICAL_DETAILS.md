# Technical Details - Song Recommendation Engine

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preparation Layer                    │
│  - CSV Loading                                               │
│  - Data Cleaning (duplicates, missing values, outliers)     │
│  - Sample Dataset Generation                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                   │
│  - Genre One-Hot Encoding                                    │
│  - Derived Features (interactions, ratios)                   │
│  - StandardScaler Normalization                              │
│  - Optional PCA Dimensionality Reduction                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Recommendation Engine Layer                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │ Cosine          │  │ K-Means         │  │ Hybrid       ││
│  │ Similarity      │  │ Clustering      │  │ Method       ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  - Streamlit Web App                                         │
│  - Command Line Demo                                         │
│  - Visualization Module                                      │
└─────────────────────────────────────────────────────────────┘
```

## Algorithm Details

### 1. Cosine Similarity

**Mathematical Formula:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||)

where:
- A, B are feature vectors
- · is dot product
- ||A|| is Euclidean norm
```

**Implementation:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity matrix (n × n)
similarity_matrix = cosine_similarity(features)

# For song i, get top k similar songs
similarities = similarity_matrix[i]
top_k_indices = np.argsort(similarities)[::-1][1:k+1]
```

**Complexity:**
- Preprocessing: O(n²) for similarity matrix
- Query: O(n log n) for sorting
- Space: O(n²) for similarity matrix

**Advantages:**
- Scale-invariant (handles different feature magnitudes)
- Interpretable scores (0-1 range)
- Fast queries after preprocessing

**Disadvantages:**
- High memory for large datasets (n² matrix)
- Doesn't capture non-linear relationships

### 2. K-Means Clustering

**Algorithm Steps:**
1. Initialize k random centroids
2. Assign each point to nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat 2-3 until convergence

**Implementation:**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

# For recommendations, find songs in same cluster
# and rank by distance to query song
```

**Complexity:**
- Training: O(n × k × i × d) where i is iterations, d is dimensions
- Query: O(n_cluster × d) where n_cluster is cluster size
- Space: O(k × d) for centroids

**Advantages:**
- Discovers natural groupings
- Efficient queries (only search within cluster)
- Captures global structure

**Disadvantages:**
- Requires choosing k (number of clusters)
- Sensitive to initialization
- Assumes spherical clusters

**Silhouette Score:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
- a(i) = avg distance to points in same cluster
- b(i) = avg distance to points in nearest other cluster
- Range: [-1, 1], higher is better
```

### 3. Hybrid Method

**Formula:**
```
hybrid_score = α × similarity_score + β × cluster_score

where:
- α = similarity_weight (default 0.7)
- β = cluster_weight (default 0.3)
- α + β = 1
```

**Cluster Score Normalization:**
```python
# Convert distance to similarity
cluster_score = 1 - (distance / max_distance)
```

## Feature Engineering

### Standard Features (from Spotify API)
| Feature | Range | Description |
|---------|-------|-------------|
| danceability | 0-1 | Rhythm stability, beat strength |
| energy | 0-1 | Perceptual intensity |
| loudness | -60-0 dB | Overall volume |
| speechiness | 0-1 | Presence of spoken words |
| acousticness | 0-1 | Acoustic vs electronic |
| instrumentalness | 0-1 | Vocal vs instrumental |
| liveness | 0-1 | Audience presence |
| valence | 0-1 | Musical positivity |
| tempo | 40-220 | Beats per minute |
| duration_ms | - | Track length |

### Derived Features

**1. Energy-Danceability Interaction**
```python
energy_danceability = energy × danceability
```
Captures "party-ability" of a song.

**2. Mood Score**
```python
mood_score = 0.5 × valence + 0.3 × energy + 0.2 × danceability
```
Weighted combination representing overall mood.

**3. Acoustic-Energy Ratio**
```python
acoustic_energy_ratio = acousticness / (energy + ε)
```
Distinguishes calm acoustic vs energetic electronic.

**4. Vocal-Instrumental Ratio**
```python
vocal_instrumental_ratio = (1 - instrumentalness) / (instrumentalness + ε)
```
Quantifies vocal prominence.

### Normalization

**StandardScaler:**
```python
X_scaled = (X - μ) / σ

where:
- μ = mean of feature
- σ = standard deviation
```

**Why StandardScaler?**
- Removes mean, scales to unit variance
- Preserves distribution shape
- Essential for distance-based algorithms
- Better than MinMaxScaler for features with outliers

### Dimensionality Reduction (Optional)

**PCA (Principal Component Analysis):**
```python
# Reduce d dimensions to k components
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X_scaled)
```

**Benefits:**
- Reduces computation time
- Removes noise
- Handles multicollinearity

**Trade-off:**
- Loss of interpretability
- Information loss (aim for 90%+ variance explained)

## Data Generation

The sample dataset generator uses statistical distributions to create realistic data:

```python
# Beta distribution for bounded features (0-1)
danceability ~ Beta(α=5, β=2)  # Skewed toward higher values

# Normal distribution for tempo
tempo ~ Normal(μ=120, σ=30)

# Genre-specific adjustments
if genre == 'electronic':
    energy += 0.2
    tempo += 10
```

**Genre Profiles:**
- **Electronic**: High energy, danceability, tempo
- **Classical**: High acousticness, instrumentalness
- **Hip-Hop**: High speechiness, moderate energy
- **Rock**: High energy, loudness
- **Jazz**: High acousticness, instrumentalness

## Performance Optimization

### Memory Optimization
```python
# Instead of storing full similarity matrix
# Use approximate nearest neighbors (for large datasets)
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=k, metric='cosine')
nn.fit(features)
distances, indices = nn.kneighbors(query_features)
```

### Computation Optimization
```python
# Pre-compute and cache similarity matrix
engine.compute_similarity_matrix()
engine.save_model()  # Save for future use

# Load pre-computed model
engine.load_model()
```

### Batch Processing
```python
# Get recommendations for multiple songs at once
similarities = cosine_similarity(query_features, all_features)
```

## Evaluation Metrics

### Clustering Quality

**Silhouette Score:**
- Range: [-1, 1]
- > 0.5: Good clustering
- 0.2-0.5: Acceptable
- < 0.2: Poor clustering

**Inertia (Within-cluster sum of squares):**
```python
inertia = Σ min(||x - μ_i||²)
```
Lower is better, but use elbow method to choose k.

### Recommendation Quality

**Diversity Score:**
```python
diversity = unique_genres / total_recommendations
```
Higher diversity may indicate broader recommendations.

**Genre Consistency:**
```python
consistency = same_genre_count / total_recommendations
```
Depends on use case - sometimes want consistency, sometimes diversity.

## Scalability Considerations

| Dataset Size | Recommended Approach |
|--------------|---------------------|
| < 10K songs | Full similarity matrix |
| 10K-100K | Clustering + approximate NN |
| 100K-1M | Locality-sensitive hashing |
| > 1M | Distributed computing (Spark) |

## Extension Points

### 1. Content-Based Filtering
Add lyrics analysis using TF-IDF or word embeddings.

### 2. Collaborative Filtering
Incorporate user listening history and ratings.

### 3. Deep Learning
Use autoencoders or neural networks for feature learning.

### 4. Multi-Modal
Combine audio features with album art, artist metadata.

### 5. Real-Time Updates
Implement incremental clustering for new songs.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.24.3 | Numerical operations |
| scikit-learn | 1.3.0 | ML algorithms |
| matplotlib | 3.7.2 | Static visualizations |
| seaborn | 0.12.2 | Statistical plots |
| plotly | 5.15.0 | Interactive charts |
| streamlit | 1.25.0 | Web interface |
| scipy | 1.11.1 | Scientific computing |

## File Structure

```
song-recommendation-engine/
├── data/
│   └── spotify_songs.csv          # Dataset (auto-generated)
├── src/
│   ├── data_preparation.py        # Data loading & cleaning
│   ├── feature_engineering.py     # Feature extraction
│   ├── recommendation_engine.py   # Core algorithms
│   └── visualizations.py          # Plotting functions
├── models/
│   ├── scaler.pkl                 # Saved StandardScaler
│   ├── pca.pkl                    # Saved PCA (if used)
│   └── recommendation_engine.pkl  # Saved models
├── visualizations/
│   └── *.png                      # Generated plots
├── app.py                         # Streamlit web app
├── demo.py                        # CLI demo
├── requirements.txt               # Dependencies
├── README.md                      # Project overview
├── USAGE_GUIDE.md                 # User guide
└── TECHNICAL_DETAILS.md           # This file
```
