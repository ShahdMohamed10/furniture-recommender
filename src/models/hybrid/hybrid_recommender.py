# Recommendation system/src/models/hybrid/hybrid_recommender.py

from surprise import SVDpp, Dataset, Reader
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineering

class HybridRecommender:
    def __init__(self):
        self.model = SVDpp(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02
        )
        self.reader = Reader(rating_scale=(1, 5))
        self.products_df = None
        self.interactions_df = None
        self.feature_eng = FeatureEngineering()
        self.similarity_matrix = None
        
    def fit(self, products_df, interactions_df):
        """Train the hybrid recommender"""
        # Store interactions dataframe
        self.interactions_df = interactions_df
        
        # Process product data
        self.products_df = self.feature_eng.create_text_features(products_df)
        
        # Calculate item similarity matrix
        self.similarity_matrix = self.feature_eng.calculate_item_similarity(self.products_df)
        
        # Train collaborative filtering model
        data = Dataset.load_from_df(
            interactions_df[['user_id', 'item_id', 'weighted_rating']], 
            self.reader
        )
        trainset = data.build_full_trainset()
        self.model.fit(trainset)

    def recommend(self, user_id, item_id, n_recommendations=5):
        """Get recommendations combining collaborative and content-based approaches"""
        if self.products_df is None:
            raise ValueError("Model not fitted yet!")
        
        # Get the selected item
        selected_item = self.products_df[self.products_df['item_id'] == item_id]
        if selected_item.empty:
            raise ValueError("Selected item not found!")
        
        selected_item = selected_item.iloc[0]
        
        # Create hybrid features
        hybrid_features = self.feature_eng.create_hybrid_features(
            user_id, 
            item_id, 
            self.products_df, 
            self.interactions_df,
            self.similarity_matrix
        )
        
        # Collaborative filtering predictions
        predictions = [
            (item_id, self.model.predict(user_id, item_id).est) 
            for item_id in self.products_df['item_id']
        ]
        
        # Create a dictionary of collaborative scores
        collab_scores = dict(predictions)
        
        # Add collaborative scores to features
        hybrid_features['collaborative_score'] = hybrid_features['item_id'].map(
            lambda x: collab_scores.get(x, 0)
        )
        
        # Calculate final hybrid score
        hybrid_features['hybrid_score'] = (
            0.4 * hybrid_features['collaborative_score'] +
            0.3 * hybrid_features['style_match'].astype(float) +
            0.1 * hybrid_features['style_preference_score'] +
            0.1 * hybrid_features['category_preference_score'] +
            0.1 * hybrid_features['diversity_score']
        )
        
        # Filter out the selected item
        recommendations = hybrid_features[hybrid_features['item_id'] != item_id].copy()
        
        # Sort by hybrid score
        recommendations = recommendations.sort_values('hybrid_score', ascending=False)
        
        # Ensure category diversity
        unique_categories = set()
        final_recommendations = []
        
        for item in recommendations.itertuples():
            if item.category not in unique_categories:
                final_recommendations.append(item)
                unique_categories.add(item.category)
            if len(final_recommendations) >= n_recommendations:
                break
        
        # Convert to DataFrame
        final_recommendations_df = pd.DataFrame(final_recommendations)
        
        # Select relevant columns
        columns_to_keep = [
            'item_id', 'name', 'category', 'style', 'price', 
            'hybrid_score', 'style_match', 'diversity_score'
        ]
        
        # Make sure all required columns exist
        for col in columns_to_keep:
            if col not in final_recommendations_df.columns:
                if col == 'style_match':
                    final_recommendations_df[col] = final_recommendations_df['style'] == selected_item['style']
                elif col == 'diversity_score':
                    final_recommendations_df[col] = 0.0
        
        # Return only the columns we need
        return final_recommendations_df[columns_to_keep]