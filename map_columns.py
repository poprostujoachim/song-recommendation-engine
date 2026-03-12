import pandas as pd
from pathlib import Path

def map_columns():
    """
    Maps various Spotify dataset column names to our standard format.
    Handles different CSV formats from Kaggle and other sources.
    """
    
    print("="*80)
    print("COLUMN MAPPER - Convert Any Spotify Dataset")
    print("="*80)
    
    csv_files = list(Path('data').glob('*.csv'))
    
    if not csv_files:
        print("\n❌ No CSV files found in 'data/' folder")
        print("Please place your Spotify dataset CSV in the 'data/' folder first.")
        return
    
    print("\n📁 Found CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file.name}")
    
    if len(csv_files) == 1:
        file_choice = 0
    else:
        choice = input(f"\nSelect file to process (1-{len(csv_files)}): ").strip()
        file_choice = int(choice) - 1
    
    input_file = csv_files[file_choice]
    
    print(f"\n📂 Loading {input_file.name}...")
    df = pd.read_csv(input_file)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"\n📋 Current columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    common_mappings = {
        'name': 'track_name',
        'song': 'track_name',
        'title': 'track_name',
        'track': 'track_name',
        
        'artist': 'artist_name',
        'artists': 'artist_name',
        'artist_name': 'artist_name',
        
        'track_genre': 'genre',
        'playlist_genre': 'genre',
        'genre': 'genre',
        'genres': 'genre',
        
        'track_id': 'track_id',
        'id': 'track_id',
        'song_id': 'track_id',
        
        'track_popularity': 'popularity',
        'popularity': 'popularity',
        'pop': 'popularity',
        
        'danceability': 'danceability',
        'energy': 'energy',
        'key': 'key',
        'loudness': 'loudness',
        'mode': 'mode',
        'speechiness': 'speechiness',
        'acousticness': 'acousticness',
        'instrumentalness': 'instrumentalness',
        'liveness': 'liveness',
        'valence': 'valence',
        'tempo': 'tempo',
        'duration_ms': 'duration_ms',
        'time_signature': 'time_signature',
    }
    
    print("\n🔄 Applying automatic column mapping...")
    
    mapped_columns = {}
    for old_col in df.columns:
        old_col_lower = old_col.lower().strip()
        if old_col_lower in common_mappings:
            mapped_columns[old_col] = common_mappings[old_col_lower]
            print(f"  ✓ {old_col} → {common_mappings[old_col_lower]}")
    
    if mapped_columns:
        df = df.rename(columns=mapped_columns)
    
    required_columns = [
        'track_name', 'artist_name', 'genre',
        'danceability', 'energy', 'valence', 'tempo',
        'acousticness', 'instrumentalness', 'speechiness',
        'loudness', 'liveness', 'duration_ms'
    ]
    
    print("\n📊 Checking required columns...")
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n⚠️  Missing columns: {', '.join(missing_columns)}")
        print("\nAvailable columns after mapping:")
        for col in df.columns:
            print(f"  • {col}")
    else:
        print("✓ All required columns present!")
    
    if 'track_id' not in df.columns:
        print("\n🔧 Generating track IDs...")
        df['track_id'] = [f'track_{i:06d}' for i in range(len(df))]
    
    if 'popularity' not in df.columns:
        print("\n🔧 Generating popularity scores...")
        import numpy as np
        df['popularity'] = np.random.randint(0, 101, len(df))
    
    print("\n🧹 Cleaning data...")
    initial_count = len(df)
    
    df = df.dropna(subset=['track_name', 'artist_name'])
    
    df = df.drop_duplicates(subset=['track_name', 'artist_name'])
    
    final_count = len(df)
    print(f"✓ Cleaned: {initial_count} → {final_count} songs")
    
    print("\n📊 Dataset Summary:")
    print(f"  • Total songs: {len(df)}")
    print(f"  • Unique artists: {df['artist_name'].nunique()}")
    if 'genre' in df.columns:
        print(f"  • Genres: {df['genre'].nunique()}")
        print(f"\n  Top genres:")
        for genre, count in df['genre'].value_counts().head(5).items():
            print(f"    - {genre}: {count}")
    
    print("\n🎵 Sample songs:")
    for idx, row in df.head(10).iterrows():
        genre = row.get('genre', 'Unknown')
        print(f"  • {row['track_name']} - {row['artist_name']} ({genre})")
    
    output_file = Path('data/spotify_songs.csv')
    
    if output_file.exists():
        backup = Path('data/spotify_songs_backup.csv')
        print(f"\n💾 Backing up existing file to {backup.name}...")
        import shutil
        shutil.copy(output_file, backup)
    
    print(f"\n💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print(f"✓ Processed dataset saved to: {output_file}")
    print(f"✓ Ready to use with {len(df)} real songs!")
    print("\nYou can now run:")
    print("  python recommend.py \"Song Name\"")
    print("  streamlit run app.py")
    print("="*80)

if __name__ == "__main__":
    map_columns()
