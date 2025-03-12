import pandas as pd

class DataProcessor:
    def __init__(self):
        self.interaction_weights = {
            'view': 1,
            'add_to_cart': 3,
            'purchase': 5
        }

    def process_product_data(self, products_df):
        """Process product metadata"""
        df = products_df.copy()
        
        # Combine features for content-based filtering
        df['combined_features'] = df.apply(
            lambda x: f"{x['category']} {x['style']} {x['material']} {x['color']} {x['brand']}", 
            axis=1
        )
        
        # Clean text features
        df['combined_features'] = df['combined_features'].str.lower()
        
        return df

    def process_interaction_data(self, interactions_df):
        """Process interaction data"""
        df = interactions_df.copy()
        
        # Apply interaction weights
        df['weighted_rating'] = df.apply(
            lambda x: max(x['rating'], 
                        self.interaction_weights.get(x['interaction_type'], 1)), 
            axis=1
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df