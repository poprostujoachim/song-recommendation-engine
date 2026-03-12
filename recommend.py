import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from recommendation_engine import RecommendationEngine
import argparse

def print_separator():
    print("\n" + "="*80)

def print_subseparator():
    print("-"*80)

def display_song_details(engine, song_idx):
    song = engine.data.loc[song_idx]
    print(f"\n🎵 SELECTED SONG")
    print_subseparator()
    print(f"Title:      {song['track_name']}")
    print(f"Artist:     {song['artist_name']}")
    print(f"Genre:      {song['genre']}")
    print(f"Popularity: {song['popularity']}/100")
    print(f"\nAudio Features:")
    print(f"  • Danceability:     {song['danceability']:.3f}")
    print(f"  • Energy:           {song['energy']:.3f}")
    print(f"  • Valence:          {song['valence']:.3f}")
    print(f"  • Tempo:            {song['tempo']:.1f} BPM")
    print(f"  • Acousticness:     {song['acousticness']:.3f}")
    print(f"  • Instrumentalness: {song['instrumentalness']:.3f}")
    print(f"  • Speechiness:      {song['speechiness']:.3f}")

def display_recommendations(recommendations, title, score_column):
    print(f"\n{title}")
    print_subseparator()
    
    if recommendations is None or len(recommendations) == 0:
        print("No recommendations found.")
        return
    
    for i, (idx, rec) in enumerate(recommendations.iterrows(), 1):
        score = rec.get(score_column, 0)
        print(f"\n{i:2d}. {rec['track_name']}")
        print(f"    Artist:     {rec['artist_name']}")
        print(f"    Genre:      {rec['genre']}")
        print(f"    Popularity: {rec['popularity']}/100")
        print(f"    Score:      {score:.4f}")

def find_song(engine, query):
    matches = engine.data[engine.data['track_name'].str.contains(query, case=False, na=False)]
    
    if len(matches) == 0:
        print(f"\n❌ No songs found matching '{query}'")
        return None
    
    if len(matches) == 1:
        return matches.index[0]
    
    print(f"\n🔍 Found {len(matches)} matches for '{query}':")
    print_subseparator()
    for i, (idx, row) in enumerate(matches.head(20).iterrows(), 1):
        print(f"{i:2d}. {row['track_name']} - {row['artist_name']} ({row['genre']})")
    
    while True:
        try:
            choice = input(f"\nSelect song (1-{min(20, len(matches))}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < min(20, len(matches)):
                return matches.iloc[choice_idx].name
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

def get_all_recommendations(engine, song_idx, n_recommendations=10):
    print_separator()
    display_song_details(engine, song_idx)
    
    print_separator()
    print("🔄 GENERATING RECOMMENDATIONS...")
    print_separator()
    
    print("\n[1/3] Computing similarity-based recommendations...")
    sim_recs = engine.recommend_by_similarity(song_index=song_idx, n=n_recommendations)
    
    print("[2/3] Computing cluster-based recommendations...")
    cluster_recs = engine.recommend_by_cluster(song_index=song_idx, n=n_recommendations)
    
    print("[3/3] Computing hybrid recommendations...")
    hybrid_recs = engine.recommend_hybrid(song_index=song_idx, n=n_recommendations)
    
    print_separator()
    display_recommendations(sim_recs, "📊 METHOD 1: COSINE SIMILARITY RECOMMENDATIONS", 'similarity_score')
    
    print_separator()
    display_recommendations(cluster_recs, "🎯 METHOD 2: CLUSTER-BASED RECOMMENDATIONS", 'distance')
    
    print_separator()
    display_recommendations(hybrid_recs, "⚡ METHOD 3: HYBRID RECOMMENDATIONS (Best of Both)", 'hybrid_score')
    
    print_separator()
    print("\n✅ RECOMMENDATION SUMMARY")
    print_subseparator()
    print(f"Total recommendations generated: {len(sim_recs) + len(cluster_recs) + len(hybrid_recs)}")
    print(f"Similarity method:  {len(sim_recs)} songs")
    print(f"Clustering method:  {len(cluster_recs)} songs")
    print(f"Hybrid method:      {len(hybrid_recs)} songs")
    
    song = engine.data.loc[song_idx]
    cluster_id = song['cluster']
    print(f"\nSource song cluster: {cluster_id}")
    
    cluster_profile = engine.get_cluster_profile(cluster_id)
    if cluster_profile:
        print(f"Cluster size: {cluster_profile['size']} songs")
        print(f"Top genres in cluster: {', '.join([f'{k} ({v})' for k, v in list(cluster_profile['top_genres'].items())[:3]])}")

def main():
    parser = argparse.ArgumentParser(description='Song Recommendation Engine - Get recommendations for any song')
    parser.add_argument('song', nargs='*', help='Song name to search for (optional)')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of recommendations per method (default: 10)')
    parser.add_argument('--index', type=int, help='Use song index directly instead of searching')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎵 SONG RECOMMENDATION ENGINE".center(80))
    print("="*80)
    
    print("\n⏳ Loading recommendation engine...")
    engine = RecommendationEngine()
    
    print("📂 Loading dataset...")
    engine.load_data('data/spotify_songs.csv')
    
    print("🔧 Preparing features...")
    engine.prepare_features(include_genre=True, include_derived=True)
    
    print("📐 Computing similarity matrix...")
    engine.compute_similarity_matrix()
    
    print("🎯 Training clustering model...")
    engine.train_clustering(n_clusters=10)
    
    print("\n✅ Engine ready!")
    
    if args.index is not None:
        if 0 <= args.index < len(engine.data):
            song_idx = args.index
            get_all_recommendations(engine, song_idx, args.num)
        else:
            print(f"❌ Invalid index. Must be between 0 and {len(engine.data)-1}")
        return
    
    if args.song:
        query = ' '.join(args.song)
        song_idx = find_song(engine, query)
        if song_idx is not None:
            get_all_recommendations(engine, song_idx, args.num)
    else:
        print("\n💡 INTERACTIVE MODE")
        print_subseparator()
        print("Enter song names to get recommendations, or 'quit' to exit.")
        
        while True:
            print_separator()
            query = input("\n🔍 Enter song name (or 'quit'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the Song Recommendation Engine!")
                break
            
            if not query:
                print("Please enter a song name.")
                continue
            
            song_idx = find_song(engine, query)
            if song_idx is not None:
                get_all_recommendations(engine, song_idx, args.num)

if __name__ == "__main__":
    main()
