import pandas as pd
import requests
from pathlib import Path
import sys

def download_spotify_dataset():
    """
    Downloads a real Spotify dataset from Kaggle or other sources.
    This uses a publicly available Spotify dataset with real songs.
    """
    
    print("="*80)
    print("DOWNLOADING REAL SPOTIFY DATASET")
    print("="*80)
    
    print("\n📥 Attempting to download real Spotify dataset...")
    print("\nOption 1: Using a curated Spotify dataset from GitHub...")
    
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
    
    try:
        print(f"\n⏳ Downloading from: {url}")
        print("This may take a minute...")
        
        df = pd.read_csv(url)
        
        print(f"\n✅ Successfully downloaded {len(df)} songs!")
        
        print("\n📊 Dataset Preview:")
        print("-"*80)
        print(f"Total songs: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        required_columns = {
            'track_name': 'track_name',
            'track_artist': 'artist_name',
            'playlist_genre': 'genre',
            'danceability': 'danceability',
            'energy': 'energy',
            'loudness': 'loudness',
            'speechiness': 'speechiness',
            'acousticness': 'acousticness',
            'instrumentalness': 'instrumentalness',
            'liveness': 'liveness',
            'valence': 'valence',
            'tempo': 'tempo',
            'duration_ms': 'duration_ms',
            'track_popularity': 'popularity'
        }
        
        print("\n🔄 Renaming columns to match our format...")
        df_processed = df.rename(columns=required_columns)
        
        df_processed = df_processed[[col for col in required_columns.values() if col in df_processed.columns]]
        
        if 'track_id' not in df_processed.columns:
            df_processed['track_id'] = [f'track_{i:06d}' for i in range(len(df_processed))]
        
        df_processed = df_processed.dropna(subset=['track_name', 'artist_name'])
        
        df_processed = df_processed.drop_duplicates(subset=['track_name', 'artist_name'])
        
        print(f"\n✅ Processed {len(df_processed)} unique songs")
        print(f"   Artists: {df_processed['artist_name'].nunique()}")
        print(f"   Genres: {df_processed['genre'].nunique()}")
        
        print("\n🎵 Sample songs:")
        print("-"*80)
        for idx, row in df_processed.head(10).iterrows():
            print(f"  • {row['track_name']} - {row['artist_name']} ({row['genre']})")
        
        print("\n📁 Saving to data/spotify_songs.csv...")
        Path('data').mkdir(exist_ok=True)
        df_processed.to_csv('data/spotify_songs.csv', index=False)
        
        print("\n✅ DATASET READY!")
        print("="*80)
        print(f"✓ Saved {len(df_processed)} real songs to data/spotify_songs.csv")
        print("✓ You can now use the recommendation engine with real data!")
        print("="*80)
        
        print("\n📊 Genre Distribution:")
        print(df_processed['genre'].value_counts())
        
        return df_processed
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n💡 Alternative: Manual Download Instructions")
        print("="*80)
        print("1. Visit: https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db")
        print("2. Download the CSV file")
        print("3. Place it in the 'data' folder as 'spotify_songs.csv'")
        print("4. Make sure it has these columns:")
        print("   - track_name, artist_name, genre, danceability, energy, valence, tempo, etc.")
        print("="*80)
        return None

def download_alternative_dataset():
    """
    Alternative: Download a smaller, well-formatted Spotify dataset
    """
    print("\n📥 Trying alternative dataset source...")
    
    try:
        url = "https://gist.githubusercontent.com/anonymous/raw/spotify_data.csv"
        df = pd.read_csv(url)
        return df
    except:
        print("Alternative source not available.")
        return None

if __name__ == "__main__":
    print("\n🎵 REAL SPOTIFY DATASET DOWNLOADER")
    print("="*80)
    print("\nThis script will download a real Spotify dataset with:")
    print("  • Real song names")
    print("  • Real artist names")
    print("  • Actual audio features from Spotify API")
    print("  • Multiple genres")
    print("\n" + "="*80)
    
    response = input("\nProceed with download? (y/n): ").strip().lower()
    
    if response == 'y':
        df = download_spotify_dataset()
        
        if df is not None:
            print("\n✅ SUCCESS! You can now run:")
            print("   python recommend.py \"Song Name\"")
            print("   or")
            print("   streamlit run app.py")
    else:
        print("\n❌ Download cancelled.")
