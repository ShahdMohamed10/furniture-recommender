from flask import Blueprint, jsonify, request, current_app
from api.utils.response_utils import success_response, error_response

content_bp = Blueprint('content', __name__)

@content_bp.route('/recommend/<item_id>', methods=['GET'])
def recommend_similar_items(item_id):
    try:
        # Get query parameters
        n_recommendations = request.args.get('n', default=5, type=int)
        
        # Get recommender and data from app context
        content_recommender = current_app.config['CONTENT_RECOMMENDER']
        processed_products = current_app.config['PROCESSED_PRODUCTS']
        
        # Check if item exists
        if item_id not in processed_products['item_id'].values:
            return error_response(f'Item with ID {item_id} not found', 404)
        
        # Get recommendations
        similar_items = content_recommender.recommend(
            item_id,
            n_recommendations=n_recommendations
        )
        
        # Get original item details
        original_item = processed_products[processed_products['item_id'] == item_id].iloc[0].to_dict()
        
        # Format response
        recommendations = []
        for _, item in similar_items.iterrows():
            recommendations.append({
                'item_id': item['item_id'],
                'name': item['name'],
                'category': item['category'],
                'style': item['style'],
                'price': float(item['price']),
                'similarity_score': float(item.get('similarity_score', 0))
            })
        
        response_data = {
            'original_item': {
                'item_id': original_item['item_id'],
                'name': original_item['name'],
                'category': original_item['category'],
                'style': original_item['style'],
                'price': float(original_item['price'])
            },
            'recommendations': recommendations
        }
        
        return success_response(response_data, "Content-based recommendations retrieved successfully")
        
    except Exception as e:
        return error_response(str(e), 500)