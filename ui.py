"""
UI Components module - Streamlit UI functions and components
Contains all tab definitions, display functions, and user interface logic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analytics import detect_outliers, compute_rating_stats


def display_header():
    """Display application header and title"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🎬 Netflix Movie Recommender")
        st.markdown("### AI-Powered Personalized Movie Recommendations")


def display_user_sidebar(movies):
    """
    Display sidebar with settings and filters
    
    Args:
        movies: DataFrame with movie data
        
    Returns:
        tuple: (user_id, num_recommendations, genres_filter, min_rating)
    """
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Step 1: Select User")
        user_id = st.number_input(
            "Enter User ID (1-9040)",
            min_value=1,
            max_value=9040,
            value=1,
            step=1
        )
        
        st.subheader("Step 2: Recommendation Options")
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        genres_filter = st.multiselect(
            "Filter by Genre (Optional)",
            options=sorted(movies['Genres'].str.split('|').explode().unique()),
            default=[]
        )
        
        min_rating = st.slider(
            "Minimum Average Rating",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )
    
    return user_id, num_recommendations, genres_filter, min_rating


def display_user_profile(user_info, user_ratings):
    """
    Display user profile metrics
    
    Args:
        user_info: Series with user information
        user_ratings: DataFrame with user's ratings
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Gender", user_info['Gender'].values[0])
    with col2:
        st.metric("📅 Age", int(user_info['Age'].values[0]))
    with col3:
        st.metric("💼 Occupation", int(user_info['Occupation'].values[0]))
    with col4:
        st.metric("⭐ Movies Rated", len(user_ratings))


def display_recommendations_tab(user_ratings, movies, num_recommendations, genres_filter, min_rating):
    """
    Display personalized movie recommendations tab
    
    Args:
        user_ratings: DataFrame with user's previous ratings
        movies: DataFrame with all movies
        num_recommendations: Number of recommendations to display
        genres_filter: List of genres to filter by
        min_rating: Minimum average rating filter
    """
    st.subheader("Personalized Movie Recommendations")
    
    if len(user_ratings) == 0:
        st.warning("⚠️ This user has not rated any movies yet.")
    else:
        # Get user's rated movies
        user_movie_ids = set(user_ratings['MovieID'].values)
        
        # Filter movies
        available_movies = movies[~movies['MovieID'].isin(user_movie_ids)].copy()
        
        if min_rating > 0:
            available_movies = available_movies[available_movies['avg_rating'] >= min_rating]
        
        if genres_filter:
            available_movies = available_movies[
                available_movies['Genres'].str.contains('|'.join(genres_filter), regex=True)
            ]
        
        if len(available_movies) == 0:
            st.info("ℹ️ No movies available matching your filters.")
        else:
            # Get user's preferences
            user_high_rated = user_ratings[user_ratings['Rating'] >= 4]
            user_liked_genres = user_high_rated.merge(
                movies[['MovieID', 'Genres']], on='MovieID', how='inner'
            )['Genres'].str.split('|').explode().value_counts()
            
            # Calculate personalized scores
            max_ratings = available_movies['total_ratings'].max()
            available_movies['popularity_normalized'] = (available_movies['total_ratings'] / max_ratings) if max_ratings > 0 else 0
            
            available_movies['recommendation_score'] = (
                0.3 * (available_movies['avg_rating'] / 5.0) +  # Movie quality
                0.2 * available_movies['popularity_normalized'] +  # Popularity
                0.5 * available_movies.apply(
                    lambda row: sum(1 for g in str(row['Genres']).split('|') if g in user_liked_genres.index) / max(1, len(str(row['Genres']).split('|'))), 
                    axis=1
                )  # Genre match
            )
            
            # Add slight randomization for variety
            available_movies['recommendation_score'] = available_movies['recommendation_score'] * (0.95 + 0.05 * np.random.random(len(available_movies)))
            
            # Get top recommendations
            top_recs = available_movies.nlargest(num_recommendations, 'recommendation_score')
            
            # Display recommendations
            cols = st.columns(2)
            for idx, (_, movie) in enumerate(top_recs.iterrows()):
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.subheader(f"#{idx + 1} ⭐")
                        st.write(f"**{movie['Title']}**")
                        
                        col_rating, col_pop, col_count = st.columns(3)
                        with col_rating:
                            st.metric("⭐ Rating", f"{movie['avg_rating']:.2f}/5")
                        with col_pop:
                            st.metric("📊 Popularity", f"{movie['popularity_normalized']*100:.2f}%")
                        with col_count:
                            st.metric("👥 Ratings", f"{int(movie['total_ratings'])}")
                        
                        genres = str(movie['Genres']).split('|')
                        genre_badges = ' '.join([f"`{g}`" for g in genres[:3]])
                        st.markdown(f"**Genres:** {genre_badges}")
                        
                        # Show personalization score
                        score_pct = min(100, int(movie['recommendation_score'] * 100))
                        st.markdown(f"**Match Score:** {score_pct}%")
                        st.progress(min(movie['recommendation_score'], 1.0))


def display_user_analytics_tab(user_ratings, movies):
    """Display user rating distribution and preferences"""
    st.subheader("User Rating Distribution & Preferences")
    
    if len(user_ratings) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            rating_dist = user_ratings['Rating'].value_counts().sort_index(ascending=False)
            fig_rating = go.Figure(data=[
                go.Bar(x=rating_dist.index, y=rating_dist.values, marker_color='#e50914')
            ])
            fig_rating.update_layout(
                title="Your Rating Distribution",
                xaxis_title="Rating",
                yaxis_title="Count",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col2:
            # Genre preferences
            user_rated_movies = movies[movies['MovieID'].isin(user_ratings['MovieID'])]
            genres_list = user_rated_movies['Genres'].str.split('|').explode()
            genre_counts = genres_list.value_counts().head(10)
            
            fig_genres = go.Figure(data=[
                go.Bar(x=genre_counts.values, y=genre_counts.index, orientation='h', marker_color='#221f1f')
            ])
            fig_genres.update_layout(
                title="Your Top Genres",
                xaxis_title="Count",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_genres, use_container_width=True)
        
        # Rating statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Avg Rating", f"{user_ratings['Rating'].mean():.2f}/5")
        with col2:
            st.metric("🏆 Highest Rated", f"{user_ratings['Rating'].max():.1f}")
        with col3:
            st.metric("📉 Lowest Rated", f"{user_ratings['Rating'].min():.1f}")
        with col4:
            st.metric("📈 Std Deviation", f"{user_ratings['Rating'].std():.2f}")
    else:
        st.info("No ratings to analyze.")


def display_movie_search_tab(movies):
    """Display movie search and exploration interface"""
    st.subheader("🔍 Search & Explore Movies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("Search by movie title")
    
    with col2:
        search_type = st.selectbox("Type", ["Title", "Genre"])
    
    if search_query:
        if search_type == "Title":
            search_results = movies[
                movies['Title'].str.contains(search_query, case=False, na=False)
            ].head(20)
        else:
            search_results = movies[
                movies['Genres'].str.contains(search_query, case=False, na=False)
            ].head(20)
        
        if len(search_results) > 0:
            st.info(f"Found {len(search_results)} movie(s)")
            
            for _, movie in search_results.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{movie['Title']}**")
                        st.write(f"Genres: {movie['Genres']}")
                    
                    with col2:
                        st.metric("⭐", f"{movie['avg_rating']:.2f}")
                    
                    with col3:
                        rating = st.slider(
                            "Rate",
                            0,
                            5,
                            key=f"rate_{movie['MovieID']}",
                            label_visibility="collapsed"
                        )
        else:
            st.warning("No movies found.")
    else:
        st.info("Enter a search term to find movies")


def display_system_analytics_tab(ratings, movies, users):
    """Display system overview and global analytics"""
    st.subheader("📈 System Overview & Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Users", len(users))
    with col2:
        st.metric("🎬 Total Movies", len(movies))
    with col3:
        st.metric("⭐ Total Ratings", len(ratings))
    with col4:
        st.metric("📊 Data Sparsity", "95.53%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Global rating distribution
        global_rating_dist = ratings['Rating'].value_counts().sort_index(ascending=False)
        fig_global = go.Figure(data=[
            go.Bar(x=global_rating_dist.index, y=global_rating_dist.values, marker_color='#e50914')
        ])
        fig_global.update_layout(
            title="Global Rating Distribution",
            xaxis_title="Rating",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_global, use_container_width=True)
    
    with col2:
        # Top movies globally
        top_movies_global = movies.nlargest(10, 'avg_rating')[['Title', 'avg_rating']]
        fig_top = go.Figure(data=[
            go.Bar(y=top_movies_global['Title'], x=top_movies_global['avg_rating'], 
                   orientation='h', marker_color='#221f1f')
        ])
        fig_top.update_layout(
            title="Top 10 Rated Movies",
            xaxis_title="Average Rating",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Core Technologies")
        st.markdown("""
        **Data Processing & Analysis Stack:**
        - ✓ Pandas for data manipulation
        - ✓ NumPy for numerical computations
        - ✓ SQL for database operations
        - ✓ Plotly for interactive visualizations
        """)
    
    with col2:
        st.subheader("📚 Core Analytics Features")
        st.markdown("""
        **Capabilities:**
        - Dataset: 675K+ ratings, 9K users, 3.5K movies
        - Statistical analysis (mean, std, percentiles)
        - Outlier detection (IQR method)
        - Advanced rating & anomaly analysis
        """)


def display_footer():
    """Display application footer"""
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎬 Netflix Movie Recommender System | Data Analysis & Visualization</p>
        <p>NumPy & Pandas Processing | Statistical Analysis | Interactive Plotly Visualizations</p>
    </div>
    """, unsafe_allow_html=True)
