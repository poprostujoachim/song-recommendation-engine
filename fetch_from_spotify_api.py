import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from pathlib import Path
import os

def fetch_spotify_data():
    """
    Fetches real song data from Spotify API.
    Requires Spotify API credentials.
    """
    
    print("="*80)
    print("SPOTIFY API DATA FETCHER")
    print("="*80)
    
    print("\n📋 Setup Instructions:")
    print("1. Go to https://developer.spotify.com/dashboard")
    print("2. Create an app (or use existing)")
    print("3. Copy your Client ID and Client Secret")
    print("\n" + "="*80)
    
    client_id = input("\nEnter your Spotify Client ID: ").strip()
    client_secret = input("Enter your Spotify Client Secret: ").strip()
    
    if not client_id or not client_secret:
        print("\n❌ Client ID and Secret are required!")
        return
    
    try:
        print("\n🔐 Authenticating with Spotify...")
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        print("✓ Authentication successful!")
        
        print("\n📥 Fetching playlists and tracks...")
        print("This will take several minutes...\n")
        
        all_tracks = []
        
        featured_playlists = [
            '37i9dQZEVXbMDoHDwVN2tF',  # Global Top 50
            '37i9dQZEVXbLRQDuF5jeBp',  # US Top 50
            '37i9dQZF1DXcBWIGoYBM5M',  # Today's Top Hits
            '37i9dQZF1DX0XUsuxWHRQd',  # RapCaviar
            '37i9dQZF1DX4dyzvuaRJ0n',  # mint
            '37i9dQZF1DX1lVhptIYRda',  # Hot Country
            '37i9dQZF1DX4JAvHpjipBk',  # New Music Friday
            '37i9dQZF1DX4sWSpwq3LiO',  # Peaceful Piano
            '37i9dQZF1DWXRqgorJj26U',  # Rock Classics
            '37i9dQZF1DX4UtSsGT1Sbe',  # All Out 80s
        ]
        
        for i, playlist_id in enumerate(featured_playlists, 1):
            try:
                print(f"[{i}/{len(featured_playlists)}] Fetching playlist {playlist_id}...")
                
                results = sp.playlist_tracks(playlist_id, limit=100)
                tracks = results['items']
                
                while results['next']:
                    results = sp.next(results)
                    tracks.extend(results['items'])
                
                for item in tracks:
                    if item['track'] is None:
                        continue
                    
                    track = item['track']
                    
                    track_id = track['id']
                    if not track_id:
                        continue
                    
                    audio_features = sp.audio_features(track_id)[0]
                    if not audio_features:
                        continue
                    
                    track_data = {
                        'track_id': track_id,
                        'track_name': track['name'],
                        'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                        'genre': 'pop',  # Will be updated based on artist
                        'danceability': audio_features['danceability'],
                        'energy': audio_features['energy'],
                        'key': audio_features['key'],
                        'loudness': audio_features['loudness'],
                        'mode': audio_features['mode'],
                        'speechiness': audio_features['speechiness'],
                        'acousticness': audio_features['acousticness'],
                        'instrumentalness': audio_features['instrumentalness'],
                        'liveness': audio_features['liveness'],
                        'valence': audio_features['valence'],
                        'tempo': audio_features['tempo'],
                        'duration_ms': audio_features['duration_ms'],
                        'time_signature': audio_features['time_signature'],
                        'popularity': track['popularity']
                    }
                    
                    all_tracks.append(track_data)
                
                print(f"  ✓ Fetched {len(tracks)} tracks (Total: {len(all_tracks)})")
                
            except Exception as e:
                print(f"  ⚠️  Error with playlist {playlist_id}: {e}")
                continue
        
        if not all_tracks:
            print("\n❌ No tracks fetched!")
            return
        
        print(f"\n✅ Fetched {len(all_tracks)} total tracks")
        
        print("\n🔧 Creating DataFrame...")
        df = pd.DataFrame(all_tracks)
        
        print("🧹 Removing duplicates...")
        df = df.drop_duplicates(subset=['track_id'])
        
        print(f"✓ {len(df)} unique tracks")
        
        print("\n🎵 Sample tracks:")
        for idx, row in df.head(10).iterrows():
            print(f"  • {row['track_name']} - {row['artist_name']}")
        
        print("\n💾 Saving to data/spotify_songs.csv...")
        Path('data').mkdir(exist_ok=True)
        df.to_csv('data/spotify_songs.csv', index=False)
        
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"✓ Saved {len(df)} real songs from Spotify API")
        print("✓ Ready to use!")
        print("\nYou can now run:")
        print("  python recommend.py \"Song Name\"")
        print("  streamlit run app.py")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Client ID and Secret are correct")
        print("2. Make sure spotipy is installed: pip install spotipy")
        print("3. Check your internet connection")

if __name__ == "__main__":
    try:
        import spotipy
    except ImportError:
        print("❌ Spotipy not installed!")
        print("\nInstall it with:")
        print("  pip install spotipy")
        exit(1)
    
    fetch_spotify_data()
