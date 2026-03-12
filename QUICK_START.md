# Quick Start Guide - Get Recommendations Now!

## Installation (One-Time Setup)

```bash
cd C:\Users\jwicz\CascadeProjects\song-recommendation-engine
pip install -r requirements.txt
```

## Usage - Get Recommendations

### Method 1: Direct Command Line (Fastest)

**Search for a song and get all recommendations:**
```bash
python recommend.py "Song 42"
```

**Get more recommendations (default is 10 per method):**
```bash
python recommend.py "Song 42" -n 20
```

**Use song by index (if you know it):**
```bash
python recommend.py --index 0
```

### Method 2: Interactive Mode

**Just run without arguments:**
```bash
python recommend.py
```

Then enter song names when prompted. Type 'quit' to exit.

### Method 3: Web Interface

**Launch the full dashboard:**
```bash
streamlit run app.py
```

Opens in browser with search, visualizations, and cluster analysis.

## What You Get

For any song you input, the program provides:

### 📊 Song Details
- Title, Artist, Genre
- Popularity score
- All audio features (danceability, energy, valence, tempo, etc.)

### 🎯 Three Recommendation Methods

**1. Cosine Similarity (Feature-Based)**
- Finds songs with similar audio characteristics
- Score: 0-1 (higher = more similar)
- Best for: Songs that "sound" similar

**2. Cluster-Based (Pattern Recognition)**
- Recommends from the same musical cluster
- Distance metric (lower = more similar)
- Best for: Songs in the same style/genre

**3. Hybrid (Best of Both)**
- Combines both methods (70% similarity + 30% clustering)
- Balanced recommendations
- Best for: Comprehensive suggestions

### 📈 Summary Statistics
- Total recommendations generated
- Cluster information
- Genre distribution in cluster

## Examples

```bash
# Search for a song
python recommend.py "Song 5"

# Get 15 recommendations per method
python recommend.py "Song 100" -n 15

# Interactive mode - search multiple songs
python recommend.py

# Use specific song index
python recommend.py --index 42
```

## Output Format

```
🎵 SELECTED SONG
--------------------------------------------------------------------------------
Title:      Song 42
Artist:     Artist_15
Genre:      electronic
Popularity: 75/100

Audio Features:
  • Danceability:     0.850
  • Energy:           0.920
  • Valence:          0.650
  ...

📊 METHOD 1: COSINE SIMILARITY RECOMMENDATIONS
--------------------------------------------------------------------------------
 1. Song 123
    Artist:     Artist_42
    Genre:      electronic
    Popularity: 80/100
    Score:      0.9876

[... 9 more recommendations ...]

🎯 METHOD 2: CLUSTER-BASED RECOMMENDATIONS
[... 10 recommendations ...]

⚡ METHOD 3: HYBRID RECOMMENDATIONS
[... 10 recommendations ...]

✅ RECOMMENDATION SUMMARY
Total recommendations: 30 songs
Similarity method:  10 songs
Clustering method:  10 songs
Hybrid method:      10 songs
```

## Tips

- **Partial matches work**: "Song 5" will find "Song 5", "Song 50", "Song 500", etc.
- **Multiple matches**: If multiple songs match, you'll be asked to choose
- **Case insensitive**: Search is not case-sensitive
- **Fast**: After initial loading (~5-10 seconds), recommendations are instant

## Troubleshooting

**"No songs found"**: Try a different search term or use interactive mode to browse

**Slow first run**: The engine needs to load data and compute similarity matrix (one-time per session)

**Want to see all songs**: Check `data/spotify_songs.csv` after first run
