from flask import Blueprint, jsonify, request, current_app
from api.utils.response_utils import success_response, error_response

collab_bp = Blueprint('collaborative', __name__)

@collab_bp.route('/recommend/<user_id>', methods=['GET'])
def recommend_for_user(user_id):
    try:
        # Get query parameters
        n_recommendations = request.args.get('n', default=5, type=int)
        
        # Get recommender and data from app context
        collaborative_recommender = current_app.config['COLLABORATIVE_RECOMMENDER']
        processed_products = current_app.config['PROCESSED_PRODUCTS']
        processed_interactions = current_app.config['PROCESSED_INTERACTIONS']
        
        # Check if user exists
        if user_id not in processed_interactions['user_id'].values:
            return error_response(f'User with ID {user_id} not found', 404)
        
        # Get recommendations
        personal_recs = collaborative_recommender.recommend(
            user_id=user_id,
            all_items=processed_products['item_id'].tolist(),
            n_recommendations=n_recommendations
        )
        
        # Format response
        recommendations = []
        shown_categories = set()
        
        for item_id, score in personal_recs:
            item = processed_products[processed_products['item_id'] == item_id].iloc[0].to_dict()
            
            if item['category'] not in shown_categories:
                shown_categories.add(item['category'])
                recommendations.append({
                    'item_id': item['item_id'],
                    'name': item['name'],
                    'category': item['category'],
                    'style': item['style'],
                    'price': float(item['price']),
                    'recommendation_score': float(score)
                })
        
        response_data = {
            'user_id': user_id,
            'recommendations': recommendations
        }
        
        return success_response(response_data, "Collaborative recommendations retrieved successfully")
        
    except Exception as e:
        return error_response(str(e), 500)