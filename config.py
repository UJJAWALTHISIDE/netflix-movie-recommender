"""
NETFLIX MOVIE RECOMMENDER SYSTEM - Configuration File
Configure all system parameters here
"""

# Dataset Configuration
DATASET_CONFIG = {
    'data_dir': '/home/ujjawal-mishra/Downloads/Netflix movie recommender/ml-1m',
    'db_path': '/home/ujjawal-mishra/Downloads/Netflix movie recommender/recommender_system.db',
    'output_dir': '/home/ujjawal-mishra/Downloads/Netflix movie recommender',
    'ratings_file': 'ratings.dat',
    'movies_file': 'movies.dat',
    'users_file': 'users.dat'
}

# Feature Engineering Config
FEATURE_CONFIG = {
    'embedding_dim': 50,
    'n_factors_svd': 50,
    'normalize_features': True,
    'fill_na_value': 0,
    'scale_method': 'minmax'  # or 'standard'
}

# Collaborative Filtering Config
CF_CONFIG = {
    'cf_metric': 'cosine',  # 'cosine', 'euclidean'
    'n_similar_users': 10,
    'n_similar_items': 10,
    'mf_factors': 20,
    'mf_epochs': 10,
    'mf_learning_rate': 0.01,
    'mf_lambda': 0.01
}

# Content-Based Filtering Config
CB_CONFIG = {
    'similarity_metric': 'cosine',
    'genre_weight': 0.5,
    'rating_weight': 0.3,
    'popularity_weight': 0.2,
    'feature_scaling': True
}

# XGBoost Config
XGBOOST_CONFIG = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 100,
    'random_state': 42,
    'sample_size': 500000,  # Use 500k samples for training
    'test_split': 0.2
}

# Neural Collaborative Filtering Config
NCF_CONFIG = {
    'embedding_dim': 50,
    'dense_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'batch_size': 256,
    'epochs': 5,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'sample_size': 500000,
    'optimizer': 'adam',
    'loss': 'mse'
}

# Ensemble Recommender Config
ENSEMBLE_CONFIG = {
    'model_weights': {
        'collaborative_filtering': 0.25,
        'content_based': 0.15,
        'xgboost': 0.35,
        'deep_learning': 0.25
    },
    'k': 10,  # Default number of recommendations
    'diversity': True,
    'trending': True,
    'consensus_threshold': 2  # Appear in at least 2 models
}

# Evaluation Config
EVALUATION_CONFIG = {
    'k_values': [5, 10, 20],
    'test_size': 0.2,
    'n_test_users': 100,
    'metrics': ['precision', 'recall', 'ndcg', 'map', 'coverage', 'novelty'],
    'random_state': 42
}

# Dashboard Config
DASHBOARD_CONFIG = {
    'output_format': 'html',
    'interactive': True,
    'color_palette': 'husl',
    'dpi': 300,
    'figsize': (16, 12),
    'theme': 'plotly_white'
}

# Visualization Config
VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'font_size': 12,
    'title_size': 14,
    'output_dir': 'visualizations'
}

# Recommendation Config
RECOMMENDATION_CONFIG = {
    'default_k': 10,
    'default_method': 'ensemble',
    'methods': ['collaborative_filtering', 'content_based', 'xgboost', 'deep_learning', 'ensemble'],
    'cold_start_strategy': 'popular',  # 'popular', 'random', 'demographic'
    'min_rating_threshold': 3.0
}

# Database Config
DATABASE_CONFIG = {
    'engine': 'sqlite',
    'connection_timeout': 30,
    'journal_mode': 'WAL',
    'synchronous': 'NORMAL',
    'cache_size': 10000,
    'batch_size': 100000
}

# Logging Config
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'recommender_system.log'
}

# Performance Config
PERFORMANCE_CONFIG = {
    'n_jobs': -1,  # Use all CPU cores
    'batch_processing': True,
    'cache_similarity': True,
    'cache_predictions': True,
    'use_gpu': False  # Set to True if CUDA available
}

# Hyperparameter Tuning Config
TUNING_CONFIG = {
    'grid_search': False,
    'random_search': False,
    'cv_folds': 5,
    'param_grid': {
        'xgboost': {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        },
        'ncf': {
            'embedding_dim': [32, 50, 64],
            'dense_layers': [[64, 32], [128, 64, 32], [256, 128, 64]],
            'learning_rate': [0.001, 0.005, 0.01]
        }
    }
}

# API Config (for future REST API)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'timeout': 30,
    'max_recommendations': 100,
    'rate_limit': 1000  # Requests per hour
}

def get_config(section=None):
    """Get configuration"""
    config = {
        'dataset': DATASET_CONFIG,
        'features': FEATURE_CONFIG,
        'cf': CF_CONFIG,
        'cb': CB_CONFIG,
        'xgboost': XGBOOST_CONFIG,
        'ncf': NCF_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'dashboard': DASHBOARD_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'recommendation': RECOMMENDATION_CONFIG,
        'database': DATABASE_CONFIG,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'tuning': TUNING_CONFIG,
        'api': API_CONFIG
    }
    
    if section:
        return config.get(section, {})
    return config

if __name__ == '__main__':
    print("NETFLIX RECOMMENDER SYSTEM - CONFIGURATION")
    print("=" * 80)
    
    config = get_config()
    for section, values in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in values.items():
            print(f"  {key}: {value}")
