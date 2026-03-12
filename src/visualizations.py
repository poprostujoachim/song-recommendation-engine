import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class Visualizations:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_feature_distributions(self, data, save_path=None):
        feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 
                       'valence', 'tempo']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(feature_cols):
            axes[idx].hist(data[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col.capitalize()} Distribution')
            axes[idx].set_xlabel(col.capitalize())
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature distributions to {save_path}")
        
        return fig
    
    def plot_genre_analysis(self, data, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        genre_counts = data['genre'].value_counts()
        axes[0, 0].barh(genre_counts.index, genre_counts.values)
        axes[0, 0].set_title('Songs per Genre')
        axes[0, 0].set_xlabel('Count')
        
        genre_popularity = data.groupby('genre')['popularity'].mean().sort_values(ascending=False)
        axes[0, 1].barh(genre_popularity.index, genre_popularity.values, color='coral')
        axes[0, 1].set_title('Average Popularity by Genre')
        axes[0, 1].set_xlabel('Popularity')
        
        genre_features = data.groupby('genre')[['energy', 'danceability', 'valence']].mean()
        genre_features.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Features by Genre')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        tempo_by_genre = data.groupby('genre')['tempo'].mean().sort_values(ascending=False)
        axes[1, 1].barh(tempo_by_genre.index, tempo_by_genre.values, color='lightgreen')
        axes[1, 1].set_title('Average Tempo by Genre')
        axes[1, 1].set_xlabel('BPM')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved genre analysis to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, data, save_path=None):
        feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 
                       'valence', 'tempo', 'popularity']
        
        corr_matrix = data[feature_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved correlation matrix to {save_path}")
        
        return fig
    
    def plot_cluster_analysis(self, data, features, cluster_labels, save_path=None):
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                     c=cluster_labels, cmap='tab10', alpha=0.6, s=30)
        axes[0, 0].set_title('Clusters in PCA Space')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        axes[0, 1].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Songs')
        
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        cluster_features = data_with_clusters.groupby('cluster')[['energy', 'danceability', 
                                                                   'valence', 'acousticness']].mean()
        cluster_features.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Features by Cluster')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        genre_cluster = pd.crosstab(data_with_clusters['cluster'], data_with_clusters['genre'])
        genre_cluster_pct = genre_cluster.div(genre_cluster.sum(axis=1), axis=0)
        
        sns.heatmap(genre_cluster_pct.T, annot=False, cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Genre Distribution by Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Genre')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved cluster analysis to {save_path}")
        
        return fig
    
    def plot_recommendation_comparison(self, source_song, recommendations, 
                                      feature_cols=['energy', 'danceability', 'valence'],
                                      save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(feature_cols))
        width = 0.15
        
        source_values = [source_song[col] for col in feature_cols]
        ax.bar(x - width*2, source_values, width, label='Source Song', color='red', alpha=0.8)
        
        for i, (idx, rec) in enumerate(recommendations.head(4).iterrows()):
            rec_values = [rec[col] for col in feature_cols]
            ax.bar(x + width*(i-1), rec_values, width, 
                  label=f"Rec {i+1}", alpha=0.7)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Value')
        ax.set_title('Feature Comparison: Source vs Recommendations')
        ax.set_xticks(x)
        ax.set_xticklabels([col.capitalize() for col in feature_cols])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved recommendation comparison to {save_path}")
        
        return fig

if __name__ == "__main__":
    from data_preparation import DataPreparation
    from recommendation_engine import RecommendationEngine
    
    print("Generating visualizations...")
    
    engine = RecommendationEngine()
    engine.load_data('data/spotify_songs.csv')
    engine.prepare_features()
    engine.compute_similarity_matrix()
    engine.train_clustering(n_clusters=8)
    
    viz = Visualizations()
    
    Path('visualizations').mkdir(exist_ok=True)
    
    print("\n[1/4] Feature distributions...")
    viz.plot_feature_distributions(engine.data, 'visualizations/feature_distributions.png')
    
    print("[2/4] Genre analysis...")
    viz.plot_genre_analysis(engine.data, 'visualizations/genre_analysis.png')
    
    print("[3/4] Correlation matrix...")
    viz.plot_correlation_matrix(engine.data, 'visualizations/correlation_matrix.png')
    
    print("[4/4] Cluster analysis...")
    viz.plot_cluster_analysis(engine.data, engine.features, 
                             engine.cluster_labels, 'visualizations/cluster_analysis.png')
    
    print("\n✓ All visualizations saved to 'visualizations/' directory")
