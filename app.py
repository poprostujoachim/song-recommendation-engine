import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from recommendation_engine import RecommendationEngine
from visualizations import Visualizations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Song Recommendation Engine",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_engine():
    engine = RecommendationEngine()
    engine.load_data('data/spotify_songs.csv')
    engine.prepare_features(include_genre=True, include_derived=True)
    engine.compute_similarity_matrix()
    engine.train_clustering(n_clusters=10)
    return engine

def create_feature_radar(song_data, recommendations):
    features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[song_data[f] for f in features],
        theta=[f.capitalize() for f in features],
        fill='toself',
        name='Source Song',
        line_color='red'
    ))
    
    for i, (idx, rec) in enumerate(recommendations.head(3).iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=[rec[f] for f in features],
            theta=[f.capitalize() for f in features],
            fill='toself',
            name=f"{rec['track_name'][:20]}...",
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Feature Comparison"
    )
    
    return fig

def main():
    st.title("🎵 Song Recommendation Engine")
    st.markdown("*Discover similar songs using machine learning algorithms*")
    
    with st.spinner("Loading recommendation engine..."):
        engine = load_engine()
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Find Recommendations", "📊 Dataset Explorer", "🎯 Cluster Analysis"])
    
    if page == "🏠 Home":
        st.header("Welcome to the Song Recommendation Engine!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Songs", len(engine.data))
        with col2:
            st.metric("Genres", engine.data['genre'].nunique())
        with col3:
            st.metric("Clusters", engine.data['cluster'].nunique())
        
        st.subheader("How It Works")
        
        tab1, tab2, tab3 = st.tabs(["Cosine Similarity", "Clustering", "Hybrid"])
        
        with tab1:
            st.markdown("""
            **Cosine Similarity Method**
            - Compares audio feature vectors between songs
            - Calculates similarity scores (0-1)
            - Returns most similar songs based on features like energy, danceability, tempo
            - Best for finding songs with similar audio characteristics
            """)
        
        with tab2:
            st.markdown("""
            **Clustering Method**
            - Groups similar songs using K-Means algorithm
            - Recommends songs from the same cluster
            - Considers overall song profile and patterns
            - Best for discovering songs within the same musical style
            """)
        
        with tab3:
            st.markdown("""
            **Hybrid Method**
            - Combines both similarity and clustering approaches
            - Weighted average of both methods
            - Provides balanced recommendations
            - Best for comprehensive recommendations
            """)
        
        st.subheader("Dataset Overview")
        
        genre_counts = engine.data['genre'].value_counts()
        fig = px.bar(x=genre_counts.index, y=genre_counts.values, 
                     labels={'x': 'Genre', 'y': 'Number of Songs'},
                     title='Songs by Genre')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Find Recommendations":
        st.header("Find Similar Songs")
        
        search_method = st.radio("Search by:", ["Song Name", "Random Song", "Genre"])
        
        selected_song_idx = None
        
        if search_method == "Song Name":
            song_search = st.text_input("Enter song name (partial match):", "")
            
            if song_search:
                matches = engine.data[engine.data['track_name'].str.contains(song_search, case=False, na=False)]
                
                if len(matches) > 0:
                    song_options = [f"{row['track_name']} - {row['artist_name']} ({row['genre']})" 
                                   for idx, row in matches.head(20).iterrows()]
                    selected = st.selectbox("Select a song:", song_options)
                    selected_song_idx = matches.iloc[song_options.index(selected)].name
                else:
                    st.warning(f"No songs found matching '{song_search}'")
        
        elif search_method == "Random Song":
            if st.button("🎲 Get Random Song"):
                import random
                selected_song_idx = random.choice(engine.data.index)
                st.session_state['random_song'] = selected_song_idx
            
            if 'random_song' in st.session_state:
                selected_song_idx = st.session_state['random_song']
        
        elif search_method == "Genre":
            selected_genre = st.selectbox("Select genre:", engine.data['genre'].unique())
            genre_songs = engine.data[engine.data['genre'] == selected_genre]
            
            song_options = [f"{row['track_name']} - {row['artist_name']}" 
                           for idx, row in genre_songs.head(50).iterrows()]
            selected = st.selectbox("Select a song:", song_options)
            selected_song_idx = genre_songs.iloc[song_options.index(selected)].name
        
        if selected_song_idx is not None:
            song = engine.data.loc[selected_song_idx]
            
            st.subheader("Selected Song")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### 🎵 {song['track_name']}")
                st.markdown(f"**Artist:** {song['artist_name']}")
                st.markdown(f"**Genre:** {song['genre']}")
                st.markdown(f"**Popularity:** {song['popularity']}/100")
                st.markdown(f"**Tempo:** {song['tempo']:.0f} BPM")
            
            with col2:
                features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']
                feature_values = [song[f] for f in features]
                
                fig = go.Figure(go.Bar(
                    x=[f.capitalize() for f in features],
                    y=feature_values,
                    marker_color='lightblue'
                ))
                fig.update_layout(title="Audio Features", yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Recommendations")
            
            method = st.selectbox("Recommendation Method:", 
                                 ["Cosine Similarity", "Clustering", "Hybrid"])
            n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
            
            if method == "Cosine Similarity":
                recommendations = engine.recommend_by_similarity(song_index=selected_song_idx, n=n_recommendations)
                score_col = 'similarity_score'
            elif method == "Clustering":
                recommendations = engine.recommend_by_cluster(song_index=selected_song_idx, n=n_recommendations)
                score_col = 'distance'
            else:
                recommendations = engine.recommend_hybrid(song_index=selected_song_idx, n=n_recommendations)
                score_col = 'hybrid_score'
            
            if recommendations is not None:
                display_cols = ['track_name', 'artist_name', 'genre', 'popularity', score_col]
                st.dataframe(recommendations[display_cols], use_container_width=True)
                
                st.subheader("Feature Comparison")
                
                rec_with_features = engine.data.loc[recommendations.index]
                fig = create_feature_radar(song, rec_with_features)
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📊 Dataset Explorer":
        st.header("Dataset Explorer")
        
        st.subheader("Feature Distributions")
        
        feature = st.selectbox("Select feature:", 
                              ['danceability', 'energy', 'valence', 'acousticness', 
                               'tempo', 'loudness', 'speechiness'])
        
        fig = px.histogram(engine.data, x=feature, nbins=30, 
                          title=f'{feature.capitalize()} Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Genre Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            genre_pop = engine.data.groupby('genre')['popularity'].mean().sort_values(ascending=False)
            fig = px.bar(x=genre_pop.index, y=genre_pop.values,
                        labels={'x': 'Genre', 'y': 'Average Popularity'},
                        title='Average Popularity by Genre')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            genre_tempo = engine.data.groupby('genre')['tempo'].mean().sort_values(ascending=False)
            fig = px.bar(x=genre_tempo.index, y=genre_tempo.values,
                        labels={'x': 'Genre', 'y': 'Average Tempo (BPM)'},
                        title='Average Tempo by Genre')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Correlations")
        
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                       'tempo', 'loudness', 'popularity']
        corr_matrix = engine.data[feature_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu',
                       aspect='auto')
        fig.update_layout(title='Feature Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Cluster Analysis":
        st.header("Cluster Analysis")
        
        st.subheader("Cluster Distribution")
        
        cluster_counts = engine.data['cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                    labels={'x': 'Cluster ID', 'y': 'Number of Songs'},
                    title='Songs per Cluster')
        st.plotly_chart(fig, use_container_width=True)
        
        selected_cluster = st.selectbox("Select cluster to explore:", 
                                       sorted(engine.data['cluster'].unique()))
        
        cluster_data = engine.data[engine.data['cluster'] == selected_cluster]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Songs in Cluster", len(cluster_data))
        with col2:
            st.metric("Avg Popularity", f"{cluster_data['popularity'].mean():.1f}")
        with col3:
            top_genre = cluster_data['genre'].mode()[0]
            st.metric("Top Genre", top_genre)
        
        st.subheader(f"Cluster {selected_cluster} - Genre Distribution")
        genre_dist = cluster_data['genre'].value_counts()
        fig = px.pie(values=genre_dist.values, names=genre_dist.index, 
                    title=f'Genres in Cluster {selected_cluster}')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Cluster {selected_cluster} - Average Features")
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
        cluster_features = cluster_data[feature_cols].mean()
        
        fig = go.Figure(go.Bar(
            x=[f.capitalize() for f in feature_cols],
            y=cluster_features.values,
            marker_color='lightgreen'
        ))
        fig.update_layout(title=f"Average Features - Cluster {selected_cluster}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Sample Songs from Cluster {selected_cluster}")
        sample_songs = cluster_data.sample(min(10, len(cluster_data)))
        st.dataframe(sample_songs[['track_name', 'artist_name', 'genre', 'popularity']], 
                    use_container_width=True)

if __name__ == "__main__":
    main()
