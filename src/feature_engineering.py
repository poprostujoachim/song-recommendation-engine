import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle
from pathlib import Path

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.genre_encoder = None
        self.pca = None
        
    def create_genre_features(self, data):
        genre_dummies = pd.get_dummies(data['genre'], prefix='genre')
        return genre_dummies
    
    def create_derived_features(self, data):
        derived = pd.DataFrame()
        
        derived['energy_danceability'] = data['energy'] * data['danceability']
        
        derived['mood_score'] = (data['valence'] * 0.5 + 
                                 data['energy'] * 0.3 + 
                                 data['danceability'] * 0.2)
        
        derived['acoustic_energy_ratio'] = data['acousticness'] / (data['energy'] + 0.01)
        
        derived['vocal_instrumental_ratio'] = (1 - data['instrumentalness']) / (data['instrumentalness'] + 0.01)
        
        derived['tempo_normalized'] = data['tempo'] / 200.0
        
        derived['duration_minutes'] = data['duration_ms'] / 60000.0
        
        return derived
    
    def scale_features(self, features, fit=True):
        if fit:
            scaled = self.scaler.fit_transform(features)
        else:
            scaled = self.scaler.transform(features)
        
        return scaled
    
    def apply_pca(self, features, n_components=5, fit=True):
        if fit:
            self.pca = PCA(n_components=n_components)
            transformed = self.pca.fit_transform(features)
            
            print(f"\n✓ PCA applied: {features.shape[1]} → {n_components} components")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
            print(f"Total variance explained: {sum(self.pca.explained_variance_ratio_):.2%}")
        else:
            transformed = self.pca.transform(features)
        
        return transformed
    
    def prepare_features(self, data, include_genre=True, include_derived=True, 
                        use_pca=False, n_components=5):
        feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms'
        ]
        
        features = data[feature_columns].copy()
        feature_names = feature_columns.copy()
        
        if include_derived:
            derived = self.create_derived_features(data)
            features = pd.concat([features, derived], axis=1)
            feature_names.extend(derived.columns.tolist())
            print(f"✓ Added {len(derived.columns)} derived features")
        
        if include_genre:
            genre_features = self.create_genre_features(data)
            features = pd.concat([features, genre_features], axis=1)
            feature_names.extend(genre_features.columns.tolist())
            print(f"✓ Added {len(genre_features.columns)} genre features")
        
        scaled_features = self.scale_features(features.values, fit=True)
        
        if use_pca:
            scaled_features = self.apply_pca(scaled_features, n_components=n_components, fit=True)
            feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        print(f"✓ Final feature matrix shape: {scaled_features.shape}")
        
        return scaled_features, feature_names
    
    def get_feature_importance(self, data):
        feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms'
        ]
        
        correlations = data[feature_columns].corr()
        
        variances = data[feature_columns].var()
        
        importance = pd.DataFrame({
            'feature': feature_columns,
            'variance': variances.values,
            'mean_correlation': correlations.abs().mean().values
        })
        
        importance = importance.sort_values('variance', ascending=False)
        
        return importance
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {filepath}")
    
    def save_pca(self, filepath='models/pca.pkl'):
        if self.pca is None:
            print("No PCA model to save")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pca, f)
        print(f"✓ PCA model saved to {filepath}")
    
    def load_pca(self, filepath='models/pca.pkl'):
        with open(filepath, 'rb') as f:
            self.pca = pickle.load(f)
        print(f"✓ PCA model loaded from {filepath}")

if __name__ == "__main__":
    from data_preparation import DataPreparation
    
    prep = DataPreparation()
    data = prep.load_data('data/spotify_songs.csv')
    data = prep.clean_data()
    
    fe = FeatureEngineering()
    
    features, feature_names = fe.prepare_features(
        data, 
        include_genre=True, 
        include_derived=True,
        use_pca=False
    )
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    importance = fe.get_feature_importance(data)
    print("\n" + "-"*60)
    print("Feature Importance (by variance):")
    print("-"*60)
    print(importance)
    
    fe.save_scaler()
