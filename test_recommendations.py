import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.data.data_processor import DataProcessor
from src.models.content_based.content_recommender import ContentBasedRecommender
from src.models.collaborative.collaborative_recommender import CollaborativeRecommender
from src.models.hybrid.hybrid_recommender import HybridRecommender

def get_user_purchase_history(user_id, interactions_df):
    """Get items purchased by a user"""
    user_purchases = interactions_df[
        (interactions_df['user_id'] == user_id) & 
        (interactions_df['interaction_type'] == 'purchase')
    ]
    return user_purchases['item_id'].tolist()

def get_similar_style_users(target_style, interactions_df, products_df):
    """Find users who purchased items of the same style"""
    style_items = products_df[products_df['style'] == target_style]['item_id'].tolist()
    style_purchases = interactions_df[
        (interactions_df['item_id'].isin(style_items)) & 
        (interactions_df['interaction_type'] == 'purchase')
    ]
    return style_purchases['user_id'].unique().tolist()

def get_style_based_recommendations(original_item, recommendations_df, interactions_df, products_df):
    """Get recommendations prioritizing style matching and user behavior"""
    
    # Find users who like the same style
    similar_style_users = get_similar_style_users(
        original_item['style'], 
        interactions_df, 
        products_df
    )
    
    # Get items frequently bought by users with similar style preferences
    style_user_purchases = interactions_df[
        (interactions_df['user_id'].isin(similar_style_users)) &
        (interactions_df['interaction_type'] == 'purchase')
    ]['item_id'].value_counts()
    
    filtered_items = []
    for _, item in recommendations_df.iterrows():
        # Skip if same category
        if item['category'] == original_item['category']:
            continue
            
        score = 0
        reasons = []
        
        # 1. Style Matching (Primary Priority)
        if item['style'] == original_item['style']:
            score += 2.0
            reasons.append('style_match')
        
        # 2. Bought by users with similar style preferences
        if item['item_id'] in style_user_purchases.index:
            purchase_frequency = style_user_purchases[item['item_id']]
            score += min(purchase_frequency * 0.5, 1.5)  # Cap at 1.5
            reasons.append('popular_with_style')
        
        if reasons:  # Only include if it matches at least one criterion
            filtered_items.append({
                'name': item['name'],
                'category': item['category'],
                'style': item['style'],
                'price': item['price'],
                'score': score,
                'reasons': reasons
            })
    
    # Sort by score and ensure category diversity
    filtered_items.sort(key=lambda x: x['score'], reverse=True)
    
    final_recommendations = []
    used_categories = set()
    
    for item in filtered_items:
        if item['category'] not in used_categories and len(final_recommendations) < 4:
            used_categories.add(item['category'])
            final_recommendations.append(item)
    
    return final_recommendations

def main():
    try:
        print("Loading data...")
        products_df = pd.read_csv('data/raw/product_metadata.csv')
        interactions_df = pd.read_csv('data/raw/user_interactions.csv')

        # Initialize models
        print("\nInitializing models...")
        data_processor = DataProcessor()
        content_recommender = ContentBasedRecommender()
        collaborative_recommender = CollaborativeRecommender()
        hybrid_recommender = HybridRecommender()

        # Process data
        processed_products = data_processor.process_product_data(products_df)
        processed_interactions = data_processor.process_interaction_data(interactions_df)

        # 1. Search Engine Results
        print("\n=== Search Engine Results ===")
        
        # Content-based recommendations
        print("\nFinding similar items...")
        content_recommender.fit(processed_products)
        
        # Get example item (a chair)
        example_item = processed_products[processed_products['category'] == 'Chair'].iloc[0]
        print(f"\nSelected item: {example_item['name']}")
        print(f"Category: {example_item['category']}")
        print(f"Style: {example_item['style']}")
        print(f"Price: ${example_item['price']:.2f}")
        
        similar_items = content_recommender.recommend(
            example_item['item_id'],
            n_recommendations=5
        )
        
        print("\nSimilar Items (Content-based):")
        print("-" * 80)
        for _, item in similar_items.iterrows():
            print(f"Name: {item['name']}")
            print(f"Category: {item['category']}")
            print(f"Style: {item['style']}")
            print(f"Price: ${item['price']:.2f}")
            print("-" * 80)

        # Collaborative recommendations
        print("\nFinding personalized recommendations...")
        collaborative_recommender.fit(processed_interactions)
        personal_recs = collaborative_recommender.recommend(
            user_id='user1',
            all_items=processed_products['item_id'].tolist(),
            n_recommendations=5
        )
        
        print("\nRecommended For You (Collaborative):")
        print("-" * 80)
        shown_categories = set()
        for item_id, score in personal_recs:
            item = processed_products[processed_products['item_id'] == item_id].iloc[0]
            if item['category'] not in shown_categories:
                shown_categories.add(item['category'])
                print(f"Name: {item['name']}")
                print(f"Category: {item['category']}")
                print(f"Style: {item['style']}")
                print(f"Price: ${item['price']:.2f}")
                print("-" * 80)

        # 2. Product Page - Frequently Bought Together
        print("\n=== Product Page Results ===")
        print("Finding style-matched items frequently bought together...")
        
        # Get initial recommendations using hybrid recommender
        hybrid_recommender.fit(processed_products, processed_interactions)
        initial_recommendations = hybrid_recommender.recommend(
            user_id='user1',
            item_id=example_item['item_id'],
            n_recommendations=20  # Get more recommendations for better filtering
        )
        
        # Get style-based recommendations
        final_recommendations = get_style_based_recommendations(
            example_item,
            initial_recommendations,
            processed_interactions,
            processed_products
        )
        
        print("\nFrequently Bought Together:")
        print("-" * 80)
        for item in final_recommendations:
            print(f"Name: {item['name']}")
            print(f"Category: {item['category']}")
            print(f"Style: {item['style']}")
            print(f"Price: ${item['price']:.2f}")
            
            # Show recommendation reasons
            if 'style_match' in item['reasons']:
                print("✓ Matches your item's style")
            if 'popular_with_style' in item['reasons']:
                print("✓ Popular among users who share your style preferences")
            print("-" * 80)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()