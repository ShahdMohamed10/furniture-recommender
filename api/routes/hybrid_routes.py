from flask import Blueprint, jsonify, request, current_app
from api.utils.response_utils import success_response, error_response

hybrid_bp = Blueprint('hybrid', __name__)

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
                'item_id': item['item_id'],
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

@hybrid_bp.route('/recommend/<user_id>/<item_id>', methods=['GET'])
def recommend_hybrid(user_id, item_id):
    try:
        # Get query parameters
        n_recommendations = request.args.get('n', default=4, type=int)
        
        # Get recommender and data from app context
        hybrid_recommender = current_app.config['HYBRID_RECOMMENDER']
        processed_products = current_app.config['PROCESSED_PRODUCTS']
        processed_interactions = current_app.config['PROCESSED_INTERACTIONS']
        
        # Check if user and item exist
        if user_id not in processed_interactions['user_id'].values:
            return error_response(f'User with ID {user_id} not found', 404)
            
        if item_id not in processed_products['item_id'].values:
            return error_response(f'Item with ID {item_id} not found', 404)
        
        # Get original item details
        example_item = processed_products[processed_products['item_id'] == item_id].iloc[0].to_dict()
        
        # Get initial recommendations using hybrid recommender
        initial_recommendations = hybrid_recommender.recommend(
            user_id=user_id,
            item_id=item_id,
            n_recommendations=20  # Get more recommendations for better filtering
        )
        
        # Get style-based recommendations
        final_recommendations = get_style_based_recommendations(
            example_item,
            initial_recommendations,
            processed_interactions,
            processed_products
        )
        
        # Format response
        recommendations = []
        for item in final_recommendations:
            recommendations.append({
                'item_id': item['item_id'],
                'name': item['name'],
                'category': item['category'],
                'style': item['style'],
                'price': float(item['price']),
                'score': float(item['score']),
                'reasons': item['reasons']
            })
        
        response_data = {
            'user_id': user_id,
            'original_item': {
                'item_id': example_item['item_id'],
                'name': example_item['name'],
                'category': example_item['category'],
                'style': example_item['style'],
                'price': float(example_item['price'])
            },
            'recommendations': recommendations
        }
        
        return success_response(response_data, "Hybrid recommendations retrieved successfully")
        
    except Exception as e:
        return error_response(str(e), 500)