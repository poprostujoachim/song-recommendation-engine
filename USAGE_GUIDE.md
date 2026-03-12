# Song Recommendation Engine - Usage Guide

## Quick Start

### 1. Installation

```bash
cd C:\Users\jwicz\CascadeProjects\song-recommendation-engine
pip install -r requirements.txt
```

### 2. Run the Project

#### Option A: Interactive Web Dashboard (Recommended)
```bash
streamlit run app.py
```
This will open a browser with an interactive interface where you can:
- Search for songs and get recommendations
- Explore the dataset visually
- Analyze clusters
- Compare different recommendation methods

#### Option B: Command Line Demo
```bash
python demo.py
```
Interactive terminal-based demo with menu options.

#### Option C: Quick Demo
```bash
python demo.py --quick
```
Automated demo showing 3 random examples.

#### Option D: Train and Test
```bash
cd src
python recommendation_engine.py
```
Trains the models and shows sample recommendations.

## Understanding the Algorithms

### Cosine Similarity
- **What it does**: Measures the angle between feature vectors
- **Best for**: Finding songs with similar audio characteristics
- **Output**: Similarity score (0-1, higher is more similar)

### K-Means Clustering
- **What it does**: Groups songs into clusters based on features
- **Best for**: Discovering songs within the same musical style
- **Output**: Distance to cluster centroid (lower is more similar)

### Hybrid Method
- **What it does**: Combines both approaches with weighted average
- **Default weights**: 70% similarity, 30% clustering
- **Best for**: Balanced, comprehensive recommendations

## Features Used

### Primary Features
- **Danceability**: How suitable for dancing (0-1)
- **Energy**: Intensity and activity (0-1)
- **Valence**: Musical positivity (0-1)
- **Tempo**: Speed in BPM
- **Acousticness**: Acoustic vs electronic (0-1)
- **Instrumentalness**: Vocal vs instrumental (0-1)
- **Speechiness**: Presence of spoken words (0-1)
- **Loudness**: Overall volume in dB
- **Liveness**: Presence of audience (0-1)

### Derived Features
- Energy × Danceability interaction
- Mood score (weighted combination)
- Acoustic/Energy ratio
- Vocal/Instrumental ratio

## Code Examples

### Basic Usage
```python
from src.recommendation_engine import RecommendationEngine

# Initialize and load
engine = RecommendationEngine()
engine.load_data('data/spotify_songs.csv')
engine.prepare_features()
engine.compute_similarity_matrix()
engine.train_clustering()

# Get recommendations by song name
recs = engine.recommend_by_similarity('Song 42', n=10)
print(recs)

# Get recommendations by index
recs = engine.recommend_by_similarity(song_index=42, n=10)
print(recs)
```

### Compare Methods
```python
# Similarity-based
sim_recs = engine.recommend_by_similarity(song_index=0, n=5)

# Clustering-based
cluster_recs = engine.recommend_by_cluster(song_index=0, n=5)

# Hybrid
hybrid_recs = engine.recommend_hybrid(song_index=0, n=5)
```

### Custom Hybrid Weights
```python
# More weight on similarity
recs = engine.recommend_hybrid(
    song_index=0, 
    n=10,
    similarity_weight=0.8,
    cluster_weight=0.2
)
```

### Explore Clusters
```python
# Get cluster profile
profile = engine.get_cluster_profile(cluster_id=3)
print(f"Cluster size: {profile['size']}")
print(f"Top genres: {profile['top_genres']}")
print(f"Avg features: {profile['feature_means']}")
```

## Visualizations

Generate all visualizations:
```bash
cd src
python visualizations.py
```

This creates:
- `visualizations/feature_distributions.png` - Feature histograms
- `visualizations/genre_analysis.png` - Genre statistics
- `visualizations/correlation_matrix.png` - Feature correlations
- `visualizations/cluster_analysis.png` - Cluster visualization

## Customization

### Change Number of Clusters
Edit `src/recommendation_engine.py`:
```python
engine.train_clustering(n_clusters=15)  # Default is 10
```

### Add More Features
Edit `src/data_preparation.py` to include additional features in `feature_columns`.

### Use PCA for Dimensionality Reduction
```python
engine.prepare_features(use_pca=True, n_components=5)
```

### Exclude Same Artist
```python
recs = engine.recommend_by_similarity(
    song_index=0, 
    n=10,
    exclude_same_artist=True
)
```

## Dataset Information

The project includes a sample dataset generator that creates realistic Spotify-like data with:
- 1000 songs (customizable)
- 10 genres
- 200 artists
- Genre-specific feature adjustments

To create a larger dataset:
```python
from src.data_preparation import DataPreparation

prep = DataPreparation()
prep.create_sample_dataset(n_songs=5000, output_path='data/spotify_songs.csv')
```

## Performance Tips

1. **Large datasets**: Use PCA to reduce feature dimensions
2. **Faster similarity**: Pre-compute and save similarity matrix
3. **Memory**: Reduce number of clusters for large datasets
4. **Speed**: Use clustering method for faster recommendations

## Troubleshooting

### "No module named 'src'"
Make sure you're running from the project root directory.

### "Dataset not found"
The system will automatically create a sample dataset if none exists.

### Streamlit not opening
Check if port 8501 is available, or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

## Evaluation Metrics

The system provides:
- **Silhouette Score**: Measures clustering quality (-1 to 1, higher is better)
- **Feature Variance**: Identifies most important features
- **Cluster Distribution**: Shows balance of clusters

## Next Steps

1. **Use Real Data**: Replace sample data with actual Spotify dataset
2. **Add More Features**: Include lyrics, release year, etc.
3. **Collaborative Filtering**: Combine with user behavior data
4. **Deep Learning**: Implement neural network embeddings
5. **API Integration**: Connect to Spotify API for real-time data
