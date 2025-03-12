# api/app.py
from flask import Flask, jsonify
from flask_cors import CORS
import os
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.data.data_processor import DataProcessor
from src.models.content_based.content_recommender import ContentBasedRecommender
from src.models.collaborative.collaborative_recommender import CollaborativeRecommender
from src.models.hybrid.hybrid_recommender import HybridRecommender
from api.routes.content_routes import content_bp
from api.routes.collab_routes import collab_bp
from api.routes.hybrid_routes import hybrid_bp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and process data
print("Loading data...")
products_df = pd.read_csv('data/raw/product_metadata.csv')
interactions_df = pd.read_csv('data/raw/user_interactions.csv')

# Initialize models
print("Initializing models...")
data_processor = DataProcessor()
processed_products = data_processor.process_product_data(products_df)
processed_interactions = data_processor.process_interaction_data(interactions_df)

# Initialize recommenders
print("Training content-based recommender...")
content_recommender = ContentBasedRecommender()
content_recommender.fit(processed_products)

print("Training collaborative recommender...")
collaborative_recommender = CollaborativeRecommender()
collaborative_recommender.fit(processed_interactions)

print("Training hybrid recommender...")
hybrid_recommender = HybridRecommender()
hybrid_recommender.fit(processed_products, processed_interactions)

# Store models and data in app context
app.config['PROCESSED_PRODUCTS'] = processed_products
app.config['PROCESSED_INTERACTIONS'] = processed_interactions
app.config['CONTENT_RECOMMENDER'] = content_recommender
app.config['COLLABORATIVE_RECOMMENDER'] = collaborative_recommender
app.config['HYBRID_RECOMMENDER'] = hybrid_recommender

# Register blueprints
app.register_blueprint(content_bp, url_prefix='/api/content')
app.register_blueprint(collab_bp, url_prefix='/api/collaborative')
app.register_blueprint(hybrid_bp, url_prefix='/api/hybrid')

@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'Furniture Recommendation API is running',
        'endpoints': {
            'content_based': '/api/content/recommend/<item_id>',
            'collaborative': '/api/collaborative/recommend/<user_id>',
            'hybrid': '/api/hybrid/recommend/<user_id>/<item_id>'
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)