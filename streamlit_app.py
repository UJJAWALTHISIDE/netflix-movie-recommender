"""
Netflix Movie Recommender System - Data Analysis & Visualization
NumPy, Pandas, SQL, and ML Fundamentals in Action

Modular Architecture:
- analytics.py: Data loading and statistical computations
- ui.py: Streamlit UI components and display logic
- streamlit_app.py: Main orchestration and page layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from analytics import load_data, compute_rating_stats, detect_outliers, compute_recommendation_metrics
from ui import (
    display_header,
    display_user_sidebar,
    display_user_profile,
    display_recommendations_tab,
    display_user_analytics_tab,
    display_movie_search_tab,
    display_system_analytics_tab,
    display_footer
)

# Page configuration
st.set_page_config(
    page_title="🎬 Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #e50914;
        text-align: center;
    }
    h2 {
        color: #221f1f;
        border-bottom: 3px solid #e50914;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load all data
@st.cache_resource
def get_data():
    """Load and cache all required datasets."""
    return load_data()

data = get_data()
movies = data['movies']
ratings = data['ratings']
users = data['users']
user_features = data['user_features']
movie_features = data['movie_features']
interaction_matrix = data['interaction_matrix']

# ==================== PAGE LAYOUT ====================

# Display header
display_header()

st.divider()

# Display sidebar for user selection and filters
user_id, num_recommendations, genres_filter, min_rating = display_user_sidebar(movies)

st.divider()

# Get user data and validate
user_info = users[users['UserID'] == user_id]
if len(user_info) == 0:
    st.error(f"❌ User ID {user_id} not found!")
    st.stop()

user_ratings = ratings[ratings['UserID'] == user_id]

# Display user profile
display_user_profile(user_info, user_ratings)

st.divider()

# ==================== TABS ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Recommendations",
    "📊 User Analytics",
    "🔍 Movie Search",
    "📈 System Analytics",
    "🔬 Advanced Analytics"
])

# TAB 1: RECOMMENDATIONS
with tab1:
    display_recommendations_tab(user_ratings, movies, num_recommendations, genres_filter, min_rating)

# TAB 2: USER ANALYTICS
with tab2:
    display_user_analytics_tab(user_ratings, movies)

# TAB 3: MOVIE SEARCH
with tab3:
    display_movie_search_tab(movies)

# TAB 4: SYSTEM ANALYTICS
with tab4:
    display_system_analytics_tab(ratings, movies, users)

# TAB 5: ADVANCED ANALYTICS
with tab5:
    st.subheader("🔬 Data Analysis & Statistical Visualization")
    
    # Subtabs for advanced analytics
    adv1, adv2, adv3 = st.tabs([
        "📈 Statistical Analysis",
        "🔍 Outliers & Anomalies",
        "⚡ Recommendation Metrics"
    ])
    
    # ========== ADVANCED TAB 1: STATISTICAL ANALYSIS ==========
    with adv1:
        st.markdown("#### Rating Distribution Analysis (NumPy & Pandas)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Global rating statistics
            st.subheader("Global Rating Statistics")
            global_stats = compute_rating_stats(ratings['Rating'].values)
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Mean", f"{global_stats['mean']:.3f}")
                st.metric("Median", f"{global_stats['median']:.3f}")
                st.metric("Std Dev", f"{global_stats['std']:.3f}")
            with metric_col2:
                st.metric("Skewness", f"{global_stats['skew']:.3f}")
                st.metric("Kurtosis", f"{global_stats['kurtosis']:.3f}")
                st.metric("IQR", f"{global_stats['iqr']:.3f}")
            
            # Box plot
            fig_box = go.Figure([
                go.Box(y=ratings['Rating'], name='Global Ratings', marker_color='#e50914')
            ])
            fig_box.update_layout(title="Rating Distribution (Box Plot)", height=400, showlegend=False)
            st.plotly_chart(fig_box)
        
        with col2:
            # User rating statistics
            st.subheader("User Rating Statistics")
            if len(user_ratings) > 0:
                user_overall_stats = compute_rating_stats(user_ratings['Rating'].values)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("User Mean", f"{user_overall_stats['mean']:.3f}")
                    st.metric("User Median", f"{user_overall_stats['median']:.3f}")
                    st.metric("User Std Dev", f"{user_overall_stats['std']:.3f}")
                with metric_col2:
                    st.metric("User Q1", f"{user_overall_stats['q1']:.3f}")
                    st.metric("User Q3", f"{user_overall_stats['q3']:.3f}")
                    st.metric("User IQR", f"{user_overall_stats['iqr']:.3f}")
                
                # Histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=user_ratings['Rating'].values,
                    nbinsx=10,
                    name='User Ratings',
                    marker_color='#221f1f'
                ))
                fig_hist.update_layout(title="User Rating Histogram", height=400, xaxis_title="Rating", yaxis_title="Count")
                st.plotly_chart(fig_hist)
            else:
                st.info("No rating data available for this user")
        
        st.divider()
        
        # Movie rating statistics by genre
        st.markdown("#### Domain Analysis: Ratings by Genre (Using Pandas GroupBy & NumPy Aggregation)")
        
        genre_ratings = []
        for genre in ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance']:
            genre_movies = movies[movies['Genres'].str.contains(genre, na=False)]['MovieID'].unique()
            genre_user_ratings = user_ratings[user_ratings['MovieID'].isin(genre_movies)]['Rating']
            
            if len(genre_user_ratings) > 0:
                genre_ratings.append({
                    'Genre': genre,
                    'Count': len(genre_user_ratings),
                    'Mean': genre_user_ratings.mean(),
                    'Std': genre_user_ratings.std(),
                    'Min': genre_user_ratings.min(),
                    'Max': genre_user_ratings.max()
                })
        
        if genre_ratings:
            genre_df = pd.DataFrame(genre_ratings)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(genre_df)
            
            with col2:
                # Violin plot
                fig_violin = go.Figure()
                for genre in genre_df['Genre']:
                    genre_movies = movies[movies['Genres'].str.contains(genre, na=False)]['MovieID'].unique()
                    genre_user_ratings = user_ratings[user_ratings['MovieID'].isin(genre_movies)]['Rating']
                    if len(genre_user_ratings) > 0:
                        fig_violin.add_trace(go.Violin(y=genre_user_ratings, name=genre, side='positive'))
                
                fig_violin.update_layout(title="Rating Distribution by Genre (Violin Plot)", height=400, yaxis_title="Rating")
                st.plotly_chart(fig_violin)
    
    # ========== ADVANCED TAB 2: OUTLIERS & ANOMALIES ==========
    with adv2:
        st.markdown("#### Outlier Detection & Anomaly Analysis (IQR & Statistical Methods)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Outliers (User Level)")
            
            if len(user_ratings) > 0:
                # Detect outliers in user ratings
                outlier_mask = detect_outliers(user_ratings['Rating'])
                outlier_count = outlier_mask.sum()
                
                st.metric("Detected Outliers", int(outlier_count))
                st.metric("Outlier Percentage", f"{(outlier_count/len(user_ratings))*100:.2f}%")
                
                if outlier_count > 0:
                    # Show outlier distribution
                    fig_outlier = go.Figure()
                    fig_outlier.add_trace(go.Histogram(
                        x=user_ratings[~outlier_mask]['Rating'],
                        name='Normal Ratings',
                        marker_color='#221f1f',
                        opacity=0.7
                    ))
                    fig_outlier.add_trace(go.Histogram(
                        x=user_ratings[outlier_mask]['Rating'],
                        name='Outlier Ratings',
                        marker_color='#e50914'
                    ))
                    
                    fig_outlier.update_layout(
                        title="Rating Distribution with Outliers Highlighted",
                        barmode='overlay',
                        height=400,
                        xaxis_title="Rating",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_outlier)
                else:
                    st.info("No outliers detected in user ratings")
            else:
                st.warning("No rating data available for this user")
        
        with col2:
            st.subheader("Movie Rating Variance Analysis")
            
            # Calculate movie rating variance
            movie_variance = ratings.groupby('MovieID')['Rating'].std().dropna()
            high_variance_movies = movie_variance.nlargest(10)
            
            fig_variance = go.Figure(data=[
                go.Bar(
                    y=movies[movies['MovieID'].isin(high_variance_movies.index)]['Title'].values,
                    x=high_variance_movies.values,
                    orientation='h',
                    marker_color='#e50914'
                )
            ])
            fig_variance.update_layout(
                title="Top 10 Movies with Highest Rating Variance",
                xaxis_title="Standard Deviation",
                height=400
            )
            st.plotly_chart(fig_variance)
        
        st.divider()
        
        st.markdown("#### Anomalous User Behavior")
        
        # Users with unusual rating patterns
        user_rating_stats = ratings.groupby('UserID')['Rating'].agg(['mean', 'std', 'count']).reset_index()
        user_rating_stats.columns = ['UserID', 'mean_rating', 'std_rating', 'num_ratings']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Very critical users
            critical_users = user_rating_stats[user_rating_stats['mean_rating'] < 2.0]
            st.metric("Very Critical Users", len(critical_users))
            if len(critical_users) > 0:
                st.dataframe(critical_users.head(5))
        
        with col2:
            # Very generous users
            generous_users = user_rating_stats[user_rating_stats['mean_rating'] > 4.0]
            st.metric("Very Generous Users", len(generous_users))
            if len(generous_users) > 0:
                st.dataframe(generous_users.head(5))
        
        with col3:
            # High variance users (inconsistent)
            inconsistent_users = user_rating_stats[user_rating_stats['std_rating'] > 1.5].dropna()
            st.metric("Inconsistent Users", len(inconsistent_users))
            if len(inconsistent_users) > 0:
                st.dataframe(inconsistent_users.head(5))
    
    # ========== ADVANCED TAB 3: RECOMMENDATION METRICS ==========
    with adv3:
        st.markdown("#### Recommendation System Quality Metrics")
        
        # Compute metrics
        metrics = compute_recommendation_metrics(ratings, movies, users)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Coverage", f"{metrics['coverage']:.2f}%", help="% of movies that can be recommended")
        with col2:
            st.metric("Sparsity", f"{metrics['sparsity']:.2f}%", help="% of missing ratings")
        with col3:
            st.metric("Avg Ratings/User", f"{metrics['avg_ratings_per_user']:.0f}", help="Average ratings per user")
        with col4:
            st.metric("Avg Rating Value", f"{metrics['avg_rating']:.2f}/5", help="Average rating across system")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Algorithm Ensemble Composition")
            
            algorithms = ['User-Based CF', 'Item-Based CF', 'Content-Based']
            weights = [0.33, 0.33, 0.34]
            
            fig_ensemble = go.Figure(data=[go.Pie(
                labels=algorithms,
                values=weights,
                marker=dict(colors=['#e50914', '#221f1f', '#FF6B6B']),
                textposition='inside',
                textinfo='label+percent'
            )])
            fig_ensemble.update_layout(title="Ensemble Model Weights", height=400)
            st.plotly_chart(fig_ensemble)
        
        with col2:
            st.markdown("#### Recommendation Diversity Analysis")
            
            # Genre diversity in recommendations
            user_genre_diversity = []
            for genre in ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance']:
                genre_movies = movies[movies['Genres'].str.contains(genre, na=False)]['MovieID'].unique()
                user_genre_count = len(user_ratings[user_ratings['MovieID'].isin(genre_movies)])
                user_genre_diversity.append({
                    'Genre': genre,
                    'Ratings': user_genre_count
                })
            
            diversity_df = pd.DataFrame(user_genre_diversity)
            
            fig_diversity = go.Figure(data=[go.Pie(
                labels=diversity_df['Genre'],
                values=diversity_df['Ratings'],
                marker=dict(colors=['#e50914', '#221f1f', '#FF6B6B', '#FFD93D', '#95E1D3']),
                textposition='inside',
                textinfo='label+percent'
            )])
            fig_diversity.update_layout(title="Your Rating Distribution by Genre", height=400)
            st.plotly_chart(fig_diversity)
        
        st.divider()
        
        st.markdown("#### Recommendation Metrics Explanation")
        
        with st.expander("📖 What do these metrics mean?"):
            st.markdown("""
            **Coverage**: The percentage of items (movies) that the system can recommend.
            Higher coverage is better as it means the system has learned something about most items.
            
            **Sparsity**: The percentage of user-item pairs that have missing ratings.
            Higher sparsity means fewer training examples, which is a common challenge in recommender systems.
            
            **Algorithm Weights**: Each algorithm contributes differently to final recommendations.
            - User-Based CF: Finds similar users and recommends their favorite items
            - Item-Based CF: Finds similar movies based on user ratings
            - Content-Based: Recommends based on movie features (genres, ratings)
            
            **Ensemble Diversity**: Recommendation diversity ensures users get varied suggestions.
            """)

# Display footer
display_footer()
