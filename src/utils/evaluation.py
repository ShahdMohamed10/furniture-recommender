import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from typing import List, Dict, Union

class RecommenderEvaluator:
    @staticmethod
    def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_precision_at_k(actual: List[str], predicted: List[str], k: int) -> float:
        """
        Calculate precision@k
        
        Args:
            actual: List of actual item IDs
            predicted: List of predicted item IDs
            k: Number of recommendations to consider
        """
        if len(predicted) > k:
            predicted = predicted[:k]
        
        if len(predicted) == 0:
            return 0.0
            
        return len(set(actual) & set(predicted)) / len(predicted)
    
    @staticmethod
    def calculate_recall_at_k(actual: List[str], predicted: List[str], k: int) -> float:
        """
        Calculate recall@k
        
        Args:
            actual: List of actual item IDs
            predicted: List of predicted item IDs
            k: Number of recommendations to consider
        """
        if len(actual) == 0:
            return 0.0
            
        if len(predicted) > k:
            predicted = predicted[:k]
            
        return len(set(actual) & set(predicted)) / len(actual)
    
    def evaluate_recommendations(self, 
                              actual: List[str], 
                              predicted: List[str], 
                              k: int = 5) -> Dict[str, float]:
        """
        Evaluate recommendations using multiple metrics
        
        Args:
            actual: List of actual item IDs
            predicted: List of predicted item IDs
            k: Number of recommendations to consider
        """
        return {
            'precision@k': self.calculate_precision_at_k(actual, predicted, k),
            'recall@k': self.calculate_recall_at_k(actual, predicted, k)
        }