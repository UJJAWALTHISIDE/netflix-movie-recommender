"""
Analytics module - All data analysis and computation functions
Contains statistical analysis, outlier detection, and metrics calculation
"""

import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data(show_spinner=False)
def compute_rating_stats(df):
    """
    Compute comprehensive rating statistics using NumPy & Pandas
    
    Args:
        df: Array of rating values
        
    Returns:
        dict: Statistical measures including mean, median, std, skew, kurtosis, quartiles, IQR
    """
    stats = {
        'mean': np.mean(df),
        'median': np.median(df),
        'std': np.std(df),
        'var': np.var(df),
        'skew': pd.Series(df).skew(),
        'kurtosis': pd.Series(df).kurtosis(),
        'q1': np.percentile(df, 25),
        'q3': np.percentile(df, 75),
        'iqr': np.percentile(df, 75) - np.percentile(df, 25)
    }
    return stats


@st.cache_data(show_spinner=False)
def compute_correlation_matrix(ratings_df, features_df):
    """
    Compute correlation between rating patterns and user features
    
    Args:
        ratings_df: DataFrame with rating data
        features_df: DataFrame with feature data
        
    Returns:
        DataFrame: Correlation matrix
    """
    merged = ratings_df.merge(features_df, on='UserID', how='inner')
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    return merged[numeric_cols].corr()


def detect_outliers(series, method='iqr'):
    """
    Detect outliers using IQR (Interquartile Range) method
    Points beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR are considered outliers
    
    Args:
        series: Pandas Series with values to check
        method: Detection method (default: 'iqr')
        
    Returns:
        Series: Boolean mask indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
    return outlier_mask


@st.cache_data(show_spinner=False)
def compute_recommendation_metrics(ratings_data, movies_df, users_df):
    """
    Compute recommendation system quality metrics
    
    Args:
        ratings_data: DataFrame with rating data
        movies_df: DataFrame with movie data
        users_df: DataFrame with user data (for total users)
        
    Returns:
        dict: Metrics including coverage, sparsity, avg ratings per user, etc.
    """
    user_id_counts = ratings_data['UserID'].value_counts()
    avg_ratings_per_user = user_id_counts.mean()
    
    # Calculate coverage: percentage of movies that have at least one rating
    movies_with_ratings = len(ratings_data['MovieID'].unique())
    total_movies = len(movies_df)
    coverage = (movies_with_ratings / total_movies) * 100
    
    return {
        'avg_ratings_per_user': avg_ratings_per_user,
        'coverage': coverage,
        'sparsity': 100 - coverage,
        'unique_users': len(user_id_counts),
        'avg_rating': ratings_data['Rating'].mean()
    }


@st.cache_data(show_spinner=False)
def load_data():
    """
    Load all required datasets from CSV files
    
    Returns:
        dict: Dictionary containing movies, ratings, users, features, interaction matrix
    """
    ratings = pd.read_csv('processed_ratings.csv')
    users = pd.read_csv('processed_users.csv')
    user_features = pd.read_csv('user_features.csv')
    movie_features = pd.read_csv('movie_features.csv')
    interaction_matrix = pd.read_csv('interaction_matrix.csv', index_col=0)
    
    # Use movie_features as main movies df since it has ratings and genres
    movies = movie_features.copy()
    
    return {
        'movies': movies,
        'ratings': ratings,
        'users': users,
        'user_features': user_features,
        'movie_features': movie_features,
        'interaction_matrix': interaction_matrix
    }
