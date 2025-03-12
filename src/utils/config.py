"""Configuration parameters"""

# Model parameters
CONTENT_BASED_PARAMS = {
    'min_df': 2,
    'max_features': 5000
}

COLLABORATIVE_PARAMS = {
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02
}

# Valid categories and styles
VALID_CATEGORIES = ['Chair', 'Table', 'Sofa', 'Bed', 'Cabinet', 'Lamp']
VALID_STYLES = ['Modern', 'Vintage', 'Industrial', 'Minimalist', 'Rustic', 'Contemporary']

# Price ranges by category
PRICE_RANGES = {
    'Chair': (100, 500),
    'Table': (200, 1000),
    'Sofa': (500, 2000),
    'Bed': (400, 1500),
    'Cabinet': (300, 1200),
    'Lamp': (50, 300)
}

# Interaction weights
INTERACTION_WEIGHTS = {
    'view': 1,
    'add_to_cart': 3,
    'purchase': 5
}