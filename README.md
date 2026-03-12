# Song Recommendation Engine

A machine learning-based music recommendation system that suggests similar songs based on audio features from Spotify dataset.

## Features

- **Cosine Similarity**: Recommends songs based on feature vector similarity
- **Clustering-Based**: Groups similar songs using K-Means clustering
- **Feature Engineering**: Analyzes genre, tempo, energy, danceability, and more
- **Interactive Dashboard**: Streamlit web interface for exploring recommendations

## Dataset

This project uses the Spotify music dataset with features including:
- Genre
- Tempo (BPM)
- Energy
- Danceability
- Valence (positivity)
- Acousticness
- Instrumentalness
- Loudness
- Speechiness

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```bash
python src/data_preparation.py
```

### 2. Train Models
```bash
python src/recommendation_engine.py
```

### 3. Run Interactive Dashboard
```bash
streamlit run app.py
```

### 4. Command Line Demo
```bash
python demo.py
```

## Project Structure

```
song-recommendation-engine/
├── data/                      # Dataset directory
├── src/
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├── feature_engineering.py # Feature extraction and scaling
│   ├── recommendation_engine.py # Similarity and clustering models
│   └── visualizations.py      # Plotting and analysis
├── models/                    # Saved models and scalers
├── app.py                     # Streamlit web interface
├── demo.py                    # Command line demo
├── requirements.txt
└── README.md
```

## How It Works

### Cosine Similarity Approach
1. Normalize audio features using StandardScaler
2. Calculate cosine similarity between song feature vectors
3. Return top N most similar songs

### Clustering Approach
1. Apply K-Means clustering to group similar songs
2. Recommend songs from the same cluster
3. Rank by distance to cluster centroid

## Example

```python
from src.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()
engine.load_data('data/spotify_songs.csv')
engine.train()

# Get recommendations
recommendations = engine.recommend_by_similarity('Song Name', n=10)
print(recommendations)
```

## Metrics & Evaluation

- Silhouette Score for clustering quality
- Feature importance analysis
- Diversity metrics for recommendations

## Future Enhancements

- Content-based filtering with lyrics
- Collaborative filtering integration
- Deep learning embeddings
- Playlist generation
