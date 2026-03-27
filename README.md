# 🎬 Netflix Movie Recommender

Movie recommendation system using machine learning.

## Files

### Code
- `streamlit_app.py` - Main web application
- `analytics.py` - Analytics functions
- `ui.py` - UI components
- `config.py` - Configuration settings

### Data
- `processed_ratings.csv` - Movie ratings data
- `processed_movies.csv` - Movie information
- `processed_users.csv` - User data
- `movie_features.csv` - Movie features
- `user_features.csv` - User features
- `interaction_matrix.csv` - User-movie interactions

## Installation

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Phase 6: XGBoost Rating Prediction
**File:** `6_xgboost_model.py`
- Predict ratings using XGBoost
- Combine user + movie features
- Feature importance analysis
- Advanced ranking

**Features Used:**
- User: mean_rating, std_rating, engagement, consistency
- Movie: avg_rating, popularity, quality, genre_count
- Interaction: rating_interaction, activity_interaction

**Performance:**
- RMSE, MAE, R² metrics
- Feature importance top 15

**Model:** `xgboost_model.bin`

---

### Phase 7: Deep Learning - NCF
**File:** `7_deep_learning_ncf.py`
- Neural Collaborative Filtering (NCF)
- Embedding layers for users & movies
- Dense layers for interaction modeling
- Rating prediction with neural networks

**Architecture:**
```
User Embedding (50 dims) ─┐
                           ├─ Concatenate ─ Dense(128) ─ Dense(64) ─ Dense(32) ─ Output [1,5]
Movie Embedding (50 dims)─┘
```

**Training:**
- Batch normalization & dropout
- Adam optimizer, MSE loss
- 5 epochs, 256 batch size

**Model:** `ncf_model.h5`

---

### Phase 8: Ensemble Recommender
**File:** `8_ensemble_recommender.py`
- Combine all 4 recommendation approaches
- Weighted scoring system
- Consensus recommendations
- Diversity & trending analysis

**Weights (Configurable):**
- Collaborative Filtering: 25%
- Content-Based: 15%
- XGBoost: 35% (best predictor)
- Deep Learning: 25%

**Recommendation Types:**
1. **Ensemble** - All models combined
2. **Consensus** - Movies from multiple models
3. **Diverse** - Different genres coverage
4. **Trending** - Recent popular movies

---

### Phase 9: Evaluation Metrics
**File:** `9_evaluation_metrics.py`
- **Precision@K** - Fraction of relevant items
- **Recall@K** - Coverage of user's preferences
- **NDCG@K** - Ranking quality (position-aware)
- **MAP@K** - Average precision at K
- **Coverage** - Catalog diversity
- **Novelty** - Recommend unpopular items
- **Diversity** - Genre & feature diversity
- **Serendipity** - Unexpected but good recommendations

**Evaluation:**
- Train/test split: 80/20
- Test on 100+ users
- Cross-model comparison

---

### Phase 10: Visualization Dashboard
**File:** `10_visualization_dashboard.py`
- Interactive Plotly dashboards
- 8+ HTML visualization files
- Statistical analysis
- Performance monitoring

**Visualizations:**
1. `00_summary_dashboard.html` - Dataset statistics
2. `01_rating_distribution.html` - Rating distribution
3. `02_user_activity.html` - User activity patterns
4. `03_movie_popularity.html` - Top 20 movies
5. `04_genre_analysis.html` - Genre distribution
6. `05_user_preferences.html` - User feature distributions
7. `06_rating_popularity.html` - Rating vs popularity scatter
8. `07_time_series.html` - Temporal patterns

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn plotly
```

### 2. Run Complete Pipeline
```bash
# Run scripts in order:
python 1_data_exploration.py
python 2_sql_database.py
python 3_feature_engineering.py
python 4_collaborative_filtering.py
python 5_content_based.py
python 6_xgboost_model.py
python 7_deep_learning_ncf.py
python 8_ensemble_recommender.py
python 9_evaluation_metrics.py
python 10_visualization_dashboard.py
```

### 3. Interactive Mode
```python
from main import NetflixRecommenderSystem

system = NetflixRecommenderSystem('/path/to/data')
system.interactive_mode()

# Options:
# 1. Get recommendations for user
# 2. Find similar movies
# 3. View top movies
# 4. View user profile
# 5. Exit
```

### 4. API Usage
```python
from main import NetflixRecommenderSystem

system = NetflixRecommenderSystem(data_dir)
system.load_all_data()

# Get recommendations
recs = system.get_recommendations(user_id=1, method='ensemble', k=10)
print(recs)

# Get similar movies
similar = system.get_similar_movies(movie_id=1, k=10)
print(similar)

# Get top movies
top = system.get_top_movies(k=15, metric='rating')
print(top)

# Get user profile
profile = system.get_user_profile(user_id=1)
```

---

## 📈 Dataset: MovieLens 1M

### Statistics
- **1,000,209 ratings** across entire dataset
- **6,040 users** with varying activity levels
- **3,706 movies** from various genres
- **Rating Scale:** 1-5 stars
- **Time Period:** 1997-2009

### Data Structure
```
Ratings:
  - UserID (1-6040)
  - MovieID (1-3883)
  - Rating (1-5)
  - Timestamp (Unix)

Movies:
  - MovieID
  - Title (with year)
  - Genres (pipe-separated)

Users:
  - UserID
  - Gender (M/F)
  - Age (18-65)
  - Occupation (21 categories)
  - ZipCode
```

---

## 🎯 Model Performance Expectations

### Rating Prediction (MAE)
- **XGBoost**: ~0.72 MAE (best)
- **Deep Learning NCF**: ~0.75 MAE
- **Matrix Factorization**: ~0.78 MAE
- **Baseline**: 1.00+ MAE

### Recommendation Quality (NDCG@10)
- **Ensemble**: 0.45-0.55
- **XGBoost CF**: 0.42-0.50
- **Content-Based**: 0.38-0.45
- **Collaborative**: 0.40-0.48

### Coverage
- **Ensemble**: 30-40% catalog coverage
- **Content-Based**: 25-35% coverage
- **Collaborative**: 20-30% coverage

---

## 🛠️ Customization

### Adjust Model Weights
```python
ensemble.set_model_weights({
    'collaborative_filtering': 0.20,
    'content_based': 0.20,
    'xgboost': 0.40,
    'deep_learning': 0.20
})
```

### Change Parameters
- **Embedding dimensions**: 30-100
- **Dense layer sizes**: 64-256
- **XGBoost max_depth**: 4-8
- **Learning rates**: 0.001-0.1

### Feature Selection
- Add more user demographics
- Incorporate temporal features
- Include metadata (year, director)
- Use context (time of day, season)

---

## 📊 File Structure

```
Netflix movie recommender/
├── ml-1m/
│   ├── ratings.dat
│   ├── movies.dat
│   ├── users.dat
│   └── README
├── 1_data_exploration.py
├── 2_sql_database.py
├── 3_feature_engineering.py
├── 4_collaborative_filtering.py
├── 5_content_based.py
├── 6_xgboost_model.py
├── 7_deep_learning_ncf.py
├── 8_ensemble_recommender.py
├── 9_evaluation_metrics.py
├── 10_visualization_dashboard.py
├── main.py
├── README.md
│
├── processed_ratings.csv
├── processed_movies.csv
├── processed_users.csv
├── user_features.csv
├── movie_features.csv
├── user_embeddings.csv
├── movie_embeddings.csv
├── user_stats.csv
├── movie_stats.csv
│
├── recommender_system.db
│
├── user_similarity.npy
├── item_similarity.npy
├── content_similarity.npy
├── mf_user_factors.npy
├── mf_item_factors.npy
│
├── xgboost_model.bin
├── ncf_model.h5
│
├── eda_analysis.png
├── 00_summary_dashboard.html
├── 01_rating_distribution.html
├── 02_user_activity.html
├── 03_movie_popularity.html
├── 04_genre_analysis.html
├── 05_user_preferences.html
├── 06_rating_popularity.html
└── 07_time_series.html
```

---

## 🔍 Advanced Features

### Hybrid Recommendations
- Combine collaborative + content-based
- Learn optimal weights from data
- Diversity constraints
- Ranking by predicted rating

### Cold Start Handling
- Content-based for new users
- Popular items for new movies
- Demographic-based for new users
- Genre recommendations for new movies

### Online Learning
- Update models with new ratings
- Incremental matrix factorization
- Real-time feature computation
- A/B testing framework

---

## 📚 Key Algorithms

### 1. Collaborative Filtering
- **User-based**: Find similar users, recommend their likes
- **Item-based**: Find similar movies, recommend related items
- **Matrix Factorization**: Decompose R = U × V^T
- **Complexity**: O(n²) for similarity, O(k·r) for estimation

### 2. Content-Based Filtering
- **Feature similarity**: Compare movie feature vectors
- **User profile**: Build user preference profile
- **Cosine similarity**: Most similar movies first
- **Complexity**: O(m·d) for features, O(m·d²) for similarity

### 3. Machine Learning
- **XGBoost**: Gradient boosting with feature importance
- **NCF**: Neural networks with embeddings
- **Regularization**: L2 penalties, dropout
- **Optimization**: Adam, SGD with momentum

---

## 🔬 Research & References

### Classical Methods
- User-based Collaborative Filtering (Resnick et al., 1994)
- Item-based CF (Sarwar et al., 2001)
- Matrix Factorization (Koren et al., 2009)

### Modern Approaches
- Neural Collaborative Filtering (He et al., 2017)
- XGBoost for Ranking (Chen & Guestrin, 2016)
- Attention Mechanisms for Recommendations
- Sequential Modeling with RNNs/Transformers

---

## ⚙️ System Requirements

### Hardware
- **RAM**: 4GB+ (8GB recommended for full pipeline)
- **Storage**: 1GB+ for datasets and models
- **CPU**: Multi-core processor

### Software
- **Python**: 3.8+
- **CUDA**: 11.0+ (optional, for GPU acceleration)
- **cuDNN**: 8.0+ (optional, for deep learning)

### Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
```

---

## 🎓 Learning Outcomes

This project demonstrates:
1. ✅ Full ML pipeline from data to deployment
2. ✅ SQL optimization and database design
3. ✅ Feature engineering & preprocessing
4. ✅ Classical collaborative filtering
5. ✅ Content-based recommendations
6. ✅ XGBoost for ranking problems
7. ✅ Deep learning embeddings
8. ✅ Ensemble methods
9. ✅ Evaluation & metrics
10. ✅ Interactive dashboards

---

## 💡 Tips for Best Results

1. **Data Quality**: Clean and preprocess thoroughly
2. **Feature Engineering**: Spend time on good features
3. **Model Tuning**: Hyperparameter optimization crucial
4. **Ensemble**: Combine diverse models
5. **Evaluation**: Use appropriate metrics for your goal
6. **A/B Testing**: Validate improvements online
7. **Monitoring**: Track model performance in production
8. **Cold Start**: Have fallback strategy for new items

---

## 🤝 Contributing

To improve this system:
1. Experiment with new features
2. Try different model architectures
3. Implement sequential models (RNNs)
4. Add attention mechanisms
5. Use implicit feedback (implicit CF)
6. Incorporate contextual information
7. Build real-time systems
8. Deploy to production

---

## 📞 Support & Questions

For issues or questions:
1. Check the relevant phase module
2. Review error messages carefully
3. Verify data files exist
4. Check dependencies are installed
5. Review configuration parameters

---

## 📝 License

This educational project demonstrates state-of-the-art recommendation systems techniques.

---

## 🎉 Conclusion

This **Netflix Movie Recommender System** provides a complete, production-ready solution combining:
- ✅ SQL databases for scalability
- ✅ Pandas/NumPy for data processing
- ✅ Scikit-learn for classical ML
- ✅ XGBoost for powerful ranking
- ✅ TensorFlow/Keras for deep learning
- ✅ Plotly for interactive analytics

**Result**: A powerful, accurate recommendation engine that combines multiple state-of-the-art techniques! 🚀

---

**Built with ❤️ using cutting-edge ML techniques**
