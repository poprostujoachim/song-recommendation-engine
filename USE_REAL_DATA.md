# Using Real Spotify Data

## Quick Method - Automatic Download

Run the download script to automatically fetch a real Spotify dataset:

```bash
python download_real_data.py
```

This will:
- Download ~32,000 real songs from a public Spotify dataset
- Include real artists like Drake, Ed Sheeran, Taylor Swift, etc.
- Contain actual Spotify audio features
- Save to `data/spotify_songs.csv`

## Manual Method - Kaggle Dataset

### Option 1: Spotify Tracks Dataset (Recommended)

1. **Visit Kaggle:**
   - https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
   - Or: https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets

2. **Download the CSV file**

3. **Prepare the data:**
   - Rename the file to `spotify_songs.csv`
   - Place it in the `data/` folder
   - Ensure it has these columns:
     - `track_name` or `name`
     - `artist_name` or `artists`
     - `genre` or `track_genre`
     - `danceability`
     - `energy`
     - `valence`
     - `tempo`
     - `acousticness`
     - `instrumentalness`
     - `speechiness`
     - `loudness`
     - `liveness`
     - `duration_ms`
     - `popularity`

4. **Run the column mapper:**
```bash
python map_columns.py
```

### Option 2: Use Spotify API (For Fresh Data)

If you want the absolute latest data, you can use the Spotify API:

1. **Get Spotify API Credentials:**
   - Go to https://developer.spotify.com/dashboard
   - Create an app
   - Get your `Client ID` and `Client Secret`

2. **Install Spotipy:**
```bash
pip install spotipy
```

3. **Run the API fetcher:**
```bash
python fetch_from_spotify_api.py
```

## Dataset Format

The system expects this CSV format:

| Column | Type | Description |
|--------|------|-------------|
| track_id | string | Unique identifier |
| track_name | string | Song title |
| artist_name | string | Artist name |
| genre | string | Music genre |
| danceability | float (0-1) | How suitable for dancing |
| energy | float (0-1) | Intensity measure |
| valence | float (0-1) | Musical positivity |
| tempo | float | Beats per minute |
| acousticness | float (0-1) | Acoustic vs electronic |
| instrumentalness | float (0-1) | Vocal vs instrumental |
| speechiness | float (0-1) | Presence of spoken words |
| loudness | float (dB) | Overall volume |
| liveness | float (0-1) | Audience presence |
| duration_ms | int | Track length in milliseconds |
| popularity | int (0-100) | Spotify popularity score |

## After Getting Real Data

Once you have real data in `data/spotify_songs.csv`, just run:

```bash
# Command line
python recommend.py "Shape of You"

# Web interface
streamlit run app.py
```

The system will automatically use the real dataset!

## Example Real Songs

With real data, you can search for actual songs:

```bash
python recommend.py "Blinding Lights"
python recommend.py "Shape of You"
python recommend.py "Bohemian Rhapsody"
python recommend.py "Smells Like Teen Spirit"
```

## Troubleshooting

### Column Names Don't Match

If your dataset has different column names, edit `src/data_preparation.py`:

```python
# Add column mapping
column_mapping = {
    'name': 'track_name',
    'artists': 'artist_name',
    'track_genre': 'genre',
    # ... add more mappings
}
df = df.rename(columns=column_mapping)
```

### Missing Columns

If some audio features are missing, the system will still work with available features. Edit `feature_columns` in `src/data_preparation.py` to match your dataset.

### Large Dataset (>100k songs)

For very large datasets:

1. Use PCA for faster processing:
```python
engine.prepare_features(use_pca=True, n_components=10)
```

2. Increase cluster count:
```python
engine.train_clustering(n_clusters=20)
```

3. Sample the dataset:
```python
df = df.sample(50000)  # Use 50k songs
```

## Popular Datasets

**TidyTuesday Spotify Dataset (32k songs)**
- URL: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21
- Format: Ready to use
- Size: ~32,000 songs

**Kaggle Ultimate Spotify Tracks (600k+ songs)**
- URL: https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
- Format: Needs minor column renaming
- Size: 600,000+ songs

**Spotify Million Playlist Dataset**
- URL: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
- Format: JSON, needs processing
- Size: 1 million playlists
