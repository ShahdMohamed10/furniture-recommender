from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.products_df = None
        self.tfidf_matrix = None

    def fit(self, products_df):
        """Train the content-based recommender"""
        self.products_df = products_df
        self.tfidf_matrix = self.tfidf.fit_transform(products_df['combined_features'])

    def recommend(self, item_id, n_recommendations=5):
        """Get similar items based on content"""
        if self.products_df is None:
            raise ValueError("Model not fitted yet!")
            
        idx = self.products_df[self.products_df['item_id'] == item_id].index[0]
        sim_scores = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix)
        sim_scores = sim_scores.flatten()
        
        # Get similar items
        item_indices = sim_scores.argsort()[::-1][1:n_recommendations+1]
        recommendations = self.products_df.iloc[item_indices].copy()
        recommendations['similarity_score'] = sim_scores[item_indices]
        
        return recommendations