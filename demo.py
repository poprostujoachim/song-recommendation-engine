import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from recommendation_engine import RecommendationEngine
import pandas as pd

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

def print_subheader(text):
    print("\n" + "-"*70)
    print(text)
    print("-"*70)

def display_song_info(data, song_index):
    song = data.loc[song_index]
    print(f"\n🎵 Song: {song['track_name']}")
    print(f"👤 Artist: {song['artist_name']}")
    print(f"🎸 Genre: {song['genre']}")
    print(f"⭐ Popularity: {song['popularity']}/100")
    print(f"\n📊 Audio Features:")
    print(f"  • Danceability: {song['danceability']:.2f}")
    print(f"  • Energy: {song['energy']:.2f}")
    print(f"  • Valence: {song['valence']:.2f}")
    print(f"  • Tempo: {song['tempo']:.0f} BPM")

def display_recommendations(recommendations, method_name):
    print_subheader(f"{method_name} - Top Recommendations")
    
    if recommendations is None or len(recommendations) == 0:
        print("No recommendations found.")
        return
    
    for i, (idx, song) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{i}. {song['track_name']} - {song['artist_name']}")
        print(f"   Genre: {song['genre']} | Popularity: {song['popularity']}/100")
        
        if 'similarity_score' in song:
            print(f"   Similarity: {song['similarity_score']:.3f}")
        elif 'distance' in song:
            print(f"   Distance: {song['distance']:.3f}")
        elif 'hybrid_score' in song:
            print(f"   Hybrid Score: {song['hybrid_score']:.3f}")

def interactive_demo(engine):
    print_header("🎵 SONG RECOMMENDATION ENGINE - INTERACTIVE DEMO 🎵")
    
    print("\nAvailable genres:")
    genres = engine.data['genre'].unique()
    for i, genre in enumerate(genres, 1):
        print(f"  {i}. {genre}")
    
    while True:
        print("\n" + "="*70)
        print("\nOptions:")
        print("  1. Search for a song by name")
        print("  2. Get random song recommendations")
        print("  3. Browse by genre")
        print("  4. Compare recommendation methods")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            song_name = input("\nEnter song name (partial match): ").strip()
            
            matches = engine.data[engine.data['track_name'].str.contains(song_name, case=False, na=False)]
            
            if len(matches) == 0:
                print(f"\n❌ No songs found matching '{song_name}'")
                continue
            
            if len(matches) > 1:
                print(f"\n🔍 Found {len(matches)} matches:")
                for i, (idx, row) in enumerate(matches.head(10).iterrows(), 1):
                    print(f"  {i}. {row['track_name']} - {row['artist_name']} ({row['genre']})")
                
                song_choice = input(f"\nSelect song (1-{min(10, len(matches))}): ").strip()
                try:
                    song_idx = matches.iloc[int(song_choice)-1].name
                except:
                    print("Invalid selection")
                    continue
            else:
                song_idx = matches.index[0]
            
            display_song_info(engine.data, song_idx)
            
            print_subheader("Getting Recommendations...")
            
            recs = engine.recommend_by_similarity(song_index=song_idx, n=5)
            display_recommendations(recs, "Cosine Similarity Method")
            
        elif choice == '2':
            import random
            song_idx = random.choice(engine.data.index)
            
            display_song_info(engine.data, song_idx)
            
            recs = engine.recommend_by_similarity(song_index=song_idx, n=5)
            display_recommendations(recs, "Recommendations")
            
        elif choice == '3':
            print("\nAvailable genres:")
            for i, genre in enumerate(genres, 1):
                print(f"  {i}. {genre}")
            
            genre_choice = input(f"\nSelect genre (1-{len(genres)}): ").strip()
            try:
                selected_genre = genres[int(genre_choice)-1]
            except:
                print("Invalid selection")
                continue
            
            genre_songs = engine.data[engine.data['genre'] == selected_genre]
            print(f"\n🎸 {selected_genre.upper()} - {len(genre_songs)} songs")
            
            sample_songs = genre_songs.sample(min(5, len(genre_songs)))
            for i, (idx, song) in enumerate(sample_songs.iterrows(), 1):
                print(f"  {i}. {song['track_name']} - {song['artist_name']}")
            
        elif choice == '4':
            import random
            song_idx = random.choice(engine.data.index)
            
            display_song_info(engine.data, song_idx)
            
            print_subheader("Method Comparison")
            
            print("\n1️⃣ Cosine Similarity Method:")
            sim_recs = engine.recommend_by_similarity(song_index=song_idx, n=5)
            for i, (idx, song) in enumerate(sim_recs.iterrows(), 1):
                print(f"  {i}. {song['track_name']} - {song['artist_name']} (score: {song['similarity_score']:.3f})")
            
            print("\n2️⃣ Clustering Method:")
            cluster_recs = engine.recommend_by_cluster(song_index=song_idx, n=5)
            for i, (idx, song) in enumerate(cluster_recs.iterrows(), 1):
                print(f"  {i}. {song['track_name']} - {song['artist_name']} (distance: {song['distance']:.3f})")
            
            print("\n3️⃣ Hybrid Method:")
            hybrid_recs = engine.recommend_hybrid(song_index=song_idx, n=5)
            for i, (idx, song) in enumerate(hybrid_recs.iterrows(), 1):
                print(f"  {i}. {song['track_name']} - {song['artist_name']} (score: {song['hybrid_score']:.3f})")
            
        elif choice == '5':
            print("\n👋 Thanks for using the Song Recommendation Engine!")
            break
        else:
            print("\n❌ Invalid choice. Please select 1-5.")

def quick_demo(engine):
    print_header("🎵 SONG RECOMMENDATION ENGINE - QUICK DEMO 🎵")
    
    import random
    demo_indices = random.sample(list(engine.data.index), 3)
    
    for demo_idx in demo_indices:
        display_song_info(engine.data, demo_idx)
        
        print_subheader("Cosine Similarity Recommendations")
        sim_recs = engine.recommend_by_similarity(song_index=demo_idx, n=5)
        display_recommendations(sim_recs, "Similarity-Based")
        
        print_subheader("Cluster-Based Recommendations")
        cluster_recs = engine.recommend_by_cluster(song_index=demo_idx, n=5)
        display_recommendations(cluster_recs, "Clustering-Based")
        
        print("\n" + "="*70)
        input("\nPress Enter to see next example...")

if __name__ == "__main__":
    print("Loading recommendation engine...")
    
    engine = RecommendationEngine()
    
    print("Loading data...")
    engine.load_data('data/spotify_songs.csv')
    
    print("Preparing features...")
    engine.prepare_features(include_genre=True, include_derived=True)
    
    print("Computing similarity matrix...")
    engine.compute_similarity_matrix()
    
    print("Training clustering model...")
    engine.train_clustering(n_clusters=10)
    
    print("\n✓ Engine ready!\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo(engine)
    else:
        interactive_demo(engine)
