# Recommendation system/src/models/collaborative/collaborative_recommender.py
from surprise import SVDpp, Dataset, Reader, model_selection
import pandas as pd
import numpy as np

class CollaborativeRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVDpp(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all
        )
        self.reader = Reader(rating_scale=(1, 5))
        self.best_params = None
        
    def tune_hyperparameters(self, interactions_df):
        """Tune hyperparameters using grid search"""
        data = Dataset.load_from_df(
            interactions_df[['user_id', 'item_id', 'weighted_rating']], 
            self.reader
        )
        
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [10, 20, 30],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.01, 0.02, 0.05]
        }
        
        gs = model_selection.GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3)
        gs.fit(data)
        
        self.best_params = gs.best_params['rmse']
        print(f"Best parameters: {self.best_params}")
        
        # Update model with best parameters
        self.model = SVDpp(
            n_factors=self.best_params['n_factors'],
            n_epochs=self.best_params['n_epochs'],
            lr_all=self.best_params['lr_all'],
            reg_all=self.best_params['reg_all']
        )
        
    def fit(self, interactions_df):
        """Train the collaborative filtering model"""
        data = Dataset.load_from_df(
            interactions_df[['user_id', 'item_id', 'weighted_rating']], 
            self.reader
        )
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
    def evaluate(self, interactions_df, k=5):
        """Evaluate the model using cross-validation"""
        data = Dataset.load_from_df(
            interactions_df[['user_id', 'item_id', 'weighted_rating']], 
            self.reader
        )
        
        # Perform cross-validation
        cv_results = model_selection.cross_validate(self.model, data, measures=['rmse', 'mae'], cv=5, verbose=False)
        
        print(f"RMSE: {np.mean(cv_results['test_rmse']):.4f}")
        print(f"MAE: {np.mean(cv_results['test_mae']):.4f}")
        
        return cv_results

    def recommend(self, user_id, all_items, n_recommendations=5):
        """Get recommendations for a user with confidence scores"""
        predictions = [
            (item_id, self.model.predict(user_id, item_id).est) 
            for item_id in all_items
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def explain_recommendation(self, user_id, item_id, interactions_df, products_df):
        """Generate an explanation for why an item was recommended"""
        # Get the user's top rated items
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        top_items = user_interactions.sort_values('weighted_rating', ascending=False)['item_id'].tolist()[:3]
        
        # Find users who also liked these items
        similar_users = interactions_df[
            (interactions_df['item_id'].isin(top_items)) & 
            (interactions_df['weighted_rating'] >= 4)
        ]['user_id'].unique()
        
        # Check if these users also liked the recommended item
        item_interactions = interactions_df[
            (interactions_df['user_id'].isin(similar_users)) & 
            (interactions_df['item_id'] == item_id)
        ]
        
        if not item_interactions.empty:
            # Get the names of the top items
            top_item_names = products_df[products_df['item_id'].isin(top_items)]['name'].tolist()
            return f"Recommended because users who liked {', '.join(top_item_names)} also liked this item."
        else:
            return "Recommended based on your overall preferences."