# Recommendation system/src/features/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureEngineering:
    def __init__(self):
        """Initialize feature engineering class"""
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=2,
            max_features=5000
        )
        self.interaction_weights = {
            'view': 1,
            'add_to_cart': 3,
            'purchase': 5
        }
        
    def create_text_features(self, products_df):
        """Create text features for content-based filtering"""
        df = products_df.copy()
        
        # Combine features for content-based filtering
        df['combined_features'] = df.apply(
            lambda x: f"{x['category']} {x['style']} {x['material']} {x['color']} {x['brand']}", 
            axis=1
        )
        
        # Clean text features
        df['combined_features'] = df['combined_features'].str.lower()
        
        return df
    
    def create_tfidf_matrix(self, products_df):
        """Create TF-IDF matrix for product features"""
        if 'combined_features' not in products_df.columns:
            products_df = self.create_text_features(products_df)
            
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(products_df['combined_features'])
        return tfidf_matrix
    
    def calculate_item_similarity(self, products_df):
        """Calculate item-item similarity matrix"""
        tfidf_matrix = self.create_tfidf_matrix(products_df)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    
    def create_user_item_matrix(self, interactions_df):
        """Create user-item interaction matrix"""
        # Apply weights to different interaction types
        weighted_df = interactions_df.copy()
        weighted_df['weighted_rating'] = weighted_df.apply(
            lambda x: max(x['rating'], 
                        self.interaction_weights.get(x['interaction_type'], 1)), 
            axis=1
        )
        
        # Create pivot table
        user_item_matrix = weighted_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='weighted_rating',
            aggfunc='mean',
            fill_value=0
        )
        
        return user_item_matrix
    
    def extract_style_preferences(self, user_id, interactions_df, products_df):
        """Extract style preferences for a user"""
        # Get user's interactions
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if user_interactions.empty:
            return {}
        
        # Get items interacted with
        interacted_items = user_interactions['item_id'].tolist()
        
        # Get styles of these items
        interacted_items_df = products_df[products_df['item_id'].isin(interacted_items)]
        
        if interacted_items_df.empty:
            return {}
        
        # Calculate style preferences
        style_counts = interacted_items_df['style'].value_counts()
        total_interactions = len(interacted_items_df)
        
        # Convert to preference scores
        style_preferences = {
            style: count / total_interactions 
            for style, count in style_counts.items()
        }
        
        return style_preferences
    
    def extract_category_preferences(self, user_id, interactions_df, products_df):
        """Extract category preferences for a user"""
        # Get user's interactions
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if user_interactions.empty:
            return {}
        
        # Get items interacted with
        interacted_items = user_interactions['item_id'].tolist()
        
        # Get categories of these items
        interacted_items_df = products_df[products_df['item_id'].isin(interacted_items)]
        
        if interacted_items_df.empty:
            return {}
        
        # Calculate category preferences
        category_counts = interacted_items_df['category'].value_counts()
        total_interactions = len(interacted_items_df)
        
        # Convert to preference scores
        category_preferences = {
            category: count / total_interactions 
            for category, count in category_counts.items()
        }
        
        return category_preferences
    
    def create_diversity_features(self, recommendations_df, original_item):
        """Create features to ensure diversity in recommendations"""
        df = recommendations_df.copy()
        
        # Add diversity score based on category difference
        df['different_category'] = df['category'] != original_item['category']
        
        # Add diversity score based on price range
        df['price_difference'] = abs(df['price'] - original_item['price']) / max(original_item['price'], 1)
        
        # Normalize price difference
        max_price_diff = df['price_difference'].max()
        if max_price_diff > 0:
            df['price_difference'] = df['price_difference'] / max_price_diff
        
        # Calculate diversity score
        df['diversity_score'] = df['different_category'].astype(int) * 0.7 + df['price_difference'] * 0.3
        
        return df
    
    def create_hybrid_features(self, user_id, item_id, products_df, interactions_df, similarity_matrix=None):
        """Create features for hybrid recommendations"""
        # Get style preferences
        style_preferences = self.extract_style_preferences(user_id, interactions_df, products_df)
        
        # Get category preferences
        category_preferences = self.extract_category_preferences(user_id, interactions_df, products_df)
        
        # Get selected item details
        selected_item = products_df[products_df['item_id'] == item_id]
        if selected_item.empty:
            raise ValueError("Selected item not found!")
        
        selected_item = selected_item.iloc[0]
        
        # Create a copy of products dataframe
        df = products_df.copy()
        
        # Add style match feature
        df['style_match'] = df['style'] == selected_item['style']
        
        # Add style preference score
        df['style_preference_score'] = df['style'].map(
            lambda x: style_preferences.get(x, 0)
        )
        
        # Add category preference score
        df['category_preference_score'] = df['category'].map(
            lambda x: category_preferences.get(x, 0)
        )
        
        # Add content similarity if similarity matrix is provided
        if similarity_matrix is not None:
            item_idx = products_df[products_df['item_id'] == item_id].index[0]
            df['content_similarity'] = similarity_matrix[item_idx]
        else:
            df['content_similarity'] = 0
        
        # Create diversity features
        df = self.create_diversity_features(df, selected_item)
        
        return df