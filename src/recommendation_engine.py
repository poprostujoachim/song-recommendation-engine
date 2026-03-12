import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from pathlib import Path
from data_preparation import DataPreparation
from feature_engineering import FeatureEngineering

class RecommendationEngine:
    def __init__(self):
        self.data = None
        self.features = None
        self.feature_names = None
        self.similarity_matrix = None
        self.kmeans = None
        self.cluster_labels = None
        self.data_prep = DataPreparation()
        self.feature_eng = FeatureEngineering()
        
    def load_data(self, filepath='data/spotify_songs.csv'):
        self.data = self.data_prep.load_data(filepath)
        self.data = self.data_prep.clean_data()
        return self.data
    
    def prepare_features(self, include_genre=True, include_derived=True, use_pca=False):
        self.features, self.feature_names = self.feature_eng.prepare_features(
            self.data,
            include_genre=include_genre,
            include_derived=include_derived,
            use_pca=use_pca
        )
        return self.features
    
    def compute_similarity_matrix(self, metric='cosine'):
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.features)
        elif metric == 'euclidean':
            distances = euclidean_distances(self.features)
            self.similarity_matrix = 1 / (1 + distances)
        
        print(f"✓ Computed {metric} similarity matrix: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def train_clustering(self, n_clusters=10, random_state=42):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.features)
        
        self.data['cluster'] = self.cluster_labels
        
        silhouette_avg = silhouette_score(self.features, self.cluster_labels)
        
        print(f"✓ K-Means clustering complete")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Silhouette score: {silhouette_avg:.3f}")
        
        print("\nCluster distribution:")
        print(pd.Series(self.cluster_labels).value_counts().sort_index())
        
        return self.cluster_labels
    
    def recommend_by_similarity(self, song_name=None, song_index=None, n=10, 
                                exclude_same_artist=False):
        if song_index is None:
            if song_name is None:
                raise ValueError("Either song_name or song_index must be provided")
            
            matches = self.data[self.data['track_name'].str.contains(song_name, case=False, na=False)]
            
            if len(matches) == 0:
                print(f"Song '{song_name}' not found in dataset")
                return None
            
            if len(matches) > 1:
                print(f"Multiple matches found for '{song_name}':")
                for idx, row in matches.head(5).iterrows():
                    print(f"  [{idx}] {row['track_name']} - {row['artist_name']}")
                song_index = matches.index[0]
                print(f"\nUsing first match: {self.data.loc[song_index, 'track_name']}")
            else:
                song_index = matches.index[0]
        
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        similarities = self.similarity_matrix[song_index]
        
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        source_artist = self.data.loc[song_index, 'artist_name']
        
        for idx in similar_indices:
            if idx == song_index:
                continue
            
            if exclude_same_artist and self.data.loc[idx, 'artist_name'] == source_artist:
                continue
            
            recommendations.append(idx)
            
            if len(recommendations) >= n:
                break
        
        result = self.data.loc[recommendations, ['track_name', 'artist_name', 'genre', 'popularity']].copy()
        result['similarity_score'] = similarities[recommendations]
        
        return result
    
    def recommend_by_cluster(self, song_name=None, song_index=None, n=10):
        if self.cluster_labels is None:
            print("Clustering not performed. Training K-Means...")
            self.train_clustering()
        
        if song_index is None:
            if song_name is None:
                raise ValueError("Either song_name or song_index must be provided")
            
            matches = self.data[self.data['track_name'].str.contains(song_name, case=False, na=False)]
            
            if len(matches) == 0:
                print(f"Song '{song_name}' not found in dataset")
                return None
            
            song_index = matches.index[0]
        
        song_cluster = self.cluster_labels[song_index]
        
        cluster_songs = self.data[self.data['cluster'] == song_cluster].index.tolist()
        
        cluster_songs = [idx for idx in cluster_songs if idx != song_index]
        
        song_features = self.features[song_index].reshape(1, -1)
        cluster_features = self.features[cluster_songs]
        
        distances = euclidean_distances(song_features, cluster_features)[0]
        
        sorted_indices = np.argsort(distances)[:n]
        recommended_indices = [cluster_songs[i] for i in sorted_indices]
        
        result = self.data.loc[recommended_indices, ['track_name', 'artist_name', 'genre', 'popularity']].copy()
        result['distance'] = distances[sorted_indices]
        result['cluster'] = song_cluster
        
        return result
    
    def get_cluster_profile(self, cluster_id):
        if self.cluster_labels is None:
            print("Clustering not performed yet")
            return None
        
        cluster_mask = self.cluster_labels == cluster_id
        cluster_data = self.data[cluster_mask]
        
        feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 
                       'valence', 'tempo']
        
        profile = {
            'size': len(cluster_data),
            'top_genres': cluster_data['genre'].value_counts().head(3).to_dict(),
            'avg_popularity': cluster_data['popularity'].mean(),
            'feature_means': cluster_data[feature_cols].mean().to_dict()
        }
        
        return profile
    
    def recommend_hybrid(self, song_name=None, song_index=None, n=10, 
                        similarity_weight=0.7, cluster_weight=0.3):
        sim_recs = self.recommend_by_similarity(song_name, song_index, n=n*2)
        cluster_recs = self.recommend_by_cluster(song_name, song_index, n=n*2)
        
        if sim_recs is None or cluster_recs is None:
            return None
        
        sim_scores = {}
        for idx, row in sim_recs.iterrows():
            sim_scores[idx] = row['similarity_score']
        
        cluster_scores = {}
        max_dist = cluster_recs['distance'].max()
        for idx, row in cluster_recs.iterrows():
            cluster_scores[idx] = 1 - (row['distance'] / max_dist)
        
        all_indices = set(sim_scores.keys()) | set(cluster_scores.keys())
        
        hybrid_scores = {}
        for idx in all_indices:
            sim_score = sim_scores.get(idx, 0)
            clust_score = cluster_scores.get(idx, 0)
            hybrid_scores[idx] = (similarity_weight * sim_score + 
                                 cluster_weight * clust_score)
        
        sorted_indices = sorted(hybrid_scores.keys(), 
                               key=lambda x: hybrid_scores[x], 
                               reverse=True)[:n]
        
        result = self.data.loc[sorted_indices, ['track_name', 'artist_name', 'genre', 'popularity']].copy()
        result['hybrid_score'] = [hybrid_scores[idx] for idx in sorted_indices]
        
        return result
    
    def save_model(self, filepath='models/recommendation_engine.pkl'):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'kmeans': self.kmeans,
            'cluster_labels': self.cluster_labels,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/recommendation_engine.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.similarity_matrix = model_data['similarity_matrix']
        self.kmeans = model_data['kmeans']
        self.cluster_labels = model_data['cluster_labels']
        self.feature_names = model_data['feature_names']
        
        if self.cluster_labels is not None and self.data is not None:
            self.data['cluster'] = self.cluster_labels
        
        print(f"✓ Model loaded from {filepath}")

if __name__ == "__main__":
    print("="*60)
    print("SONG RECOMMENDATION ENGINE - TRAINING")
    print("="*60)
    
    engine = RecommendationEngine()
    
    print("\n[1/4] Loading data...")
    engine.load_data('data/spotify_songs.csv')
    
    print("\n[2/4] Preparing features...")
    engine.prepare_features(include_genre=True, include_derived=True)
    
    print("\n[3/4] Computing similarity matrix...")
    engine.compute_similarity_matrix(metric='cosine')
    
    print("\n[4/4] Training clustering model...")
    engine.train_clustering(n_clusters=10)
    
    print("\n" + "="*60)
    print("TESTING RECOMMENDATIONS")
    print("="*60)
    
    test_song_idx = 0
    test_song = engine.data.loc[test_song_idx, 'track_name']
    test_artist = engine.data.loc[test_song_idx, 'artist_name']
    
    print(f"\nSource Song: '{test_song}' by {test_artist}")
    
    print("\n" + "-"*60)
    print("Cosine Similarity Recommendations:")
    print("-"*60)
    sim_recs = engine.recommend_by_similarity(song_index=test_song_idx, n=5)
    print(sim_recs.to_string())
    
    print("\n" + "-"*60)
    print("Cluster-Based Recommendations:")
    print("-"*60)
    cluster_recs = engine.recommend_by_cluster(song_index=test_song_idx, n=5)
    print(cluster_recs.to_string())
    
    print("\n" + "-"*60)
    print("Hybrid Recommendations:")
    print("-"*60)
    hybrid_recs = engine.recommend_hybrid(song_index=test_song_idx, n=5)
    print(hybrid_recs.to_string())
    
    print("\n[5/5] Saving models...")
    engine.save_model()
    engine.feature_eng.save_scaler()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
