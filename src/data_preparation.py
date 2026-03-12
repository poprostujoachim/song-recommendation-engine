import pandas as pd
import numpy as np
from pathlib import Path

class DataPreparation:
    def __init__(self):
        self.data = None
        self.feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms'
        ]
    
    def create_sample_dataset(self, n_songs=1000, output_path='data/spotify_songs.csv'):
        np.random.seed(42)
        
        genres = ['pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical', 
                  'r&b', 'country', 'latin', 'indie']
        
        artists = [f'Artist_{i}' for i in range(200)]
        
        data = {
            'track_id': [f'track_{i:04d}' for i in range(n_songs)],
            'track_name': [f'Song {i}' for i in range(n_songs)],
            'artist_name': np.random.choice(artists, n_songs),
            'genre': np.random.choice(genres, n_songs),
            'danceability': np.random.beta(5, 2, n_songs),
            'energy': np.random.beta(5, 2, n_songs),
            'loudness': np.random.normal(-6, 3, n_songs),
            'speechiness': np.random.beta(2, 8, n_songs),
            'acousticness': np.random.beta(2, 5, n_songs),
            'instrumentalness': np.random.beta(1, 9, n_songs),
            'liveness': np.random.beta(2, 8, n_songs),
            'valence': np.random.beta(5, 5, n_songs),
            'tempo': np.random.normal(120, 30, n_songs),
            'duration_ms': np.random.normal(210000, 60000, n_songs).astype(int),
            'popularity': np.random.randint(0, 101, n_songs)
        }
        
        df = pd.DataFrame(data)
        
        genre_adjustments = {
            'electronic': {'energy': 0.2, 'danceability': 0.15, 'tempo': 10},
            'classical': {'acousticness': 0.4, 'instrumentalness': 0.5, 'energy': -0.2},
            'jazz': {'acousticness': 0.2, 'instrumentalness': 0.3},
            'hip-hop': {'speechiness': 0.2, 'energy': 0.1},
            'rock': {'energy': 0.15, 'loudness': 2},
            'country': {'acousticness': 0.2, 'valence': 0.1}
        }
        
        for genre, adjustments in genre_adjustments.items():
            mask = df['genre'] == genre
            for feature, adjustment in adjustments.items():
                if feature in ['danceability', 'energy', 'acousticness', 
                               'instrumentalness', 'valence', 'speechiness', 'liveness']:
                    df.loc[mask, feature] = np.clip(df.loc[mask, feature] + adjustment, 0, 1)
                elif feature == 'tempo':
                    df.loc[mask, feature] = df.loc[mask, feature] + adjustment
                elif feature == 'loudness':
                    df.loc[mask, feature] = df.loc[mask, feature] + adjustment
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Created sample dataset with {n_songs} songs at {output_path}")
        
        return df
    
    def load_data(self, filepath):
        try:
            self.data = pd.read_csv(filepath)
            print(f"✓ Loaded {len(self.data)} songs from {filepath}")
            return self.data
        except FileNotFoundError:
            print(f"Dataset not found at {filepath}")
            print("Creating sample dataset...")
            self.data = self.create_sample_dataset(output_path=filepath)
            return self.data
    
    def get_data_info(self):
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        print(f"\nTotal Songs: {len(self.data)}")
        print(f"Total Artists: {self.data['artist_name'].nunique()}")
        print(f"Total Genres: {self.data['genre'].nunique()}")
        
        print("\n" + "-"*60)
        print("Genre Distribution:")
        print("-"*60)
        print(self.data['genre'].value_counts())
        
        print("\n" + "-"*60)
        print("Feature Statistics:")
        print("-"*60)
        print(self.data[self.feature_columns].describe())
        
        print("\n" + "-"*60)
        print("Missing Values:")
        print("-"*60)
        print(self.data.isnull().sum())
        
        return self.data.describe()
    
    def clean_data(self):
        if self.data is None:
            print("No data loaded.")
            return None
        
        initial_count = len(self.data)
        
        self.data = self.data.dropna()
        
        self.data = self.data.drop_duplicates(subset=['track_name', 'artist_name'])
        
        for col in ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'speechiness']:
            self.data[col] = self.data[col].clip(0, 1)
        
        self.data['tempo'] = self.data['tempo'].clip(40, 220)
        
        final_count = len(self.data)
        print(f"✓ Cleaned data: {initial_count} → {final_count} songs")
        
        return self.data
    
    def get_feature_matrix(self):
        if self.data is None:
            print("No data loaded.")
            return None
        
        return self.data[self.feature_columns].values

if __name__ == "__main__":
    prep = DataPreparation()
    
    df = prep.load_data('data/spotify_songs.csv')
    
    prep.clean_data()
    
    prep.get_data_info()
