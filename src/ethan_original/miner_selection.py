import numpy as np
import pandas as pd
import joblib

class MinerSelector:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features
        
    def select_miner(self, miner_pool):
        """
        Select the most efficient miner from a pool using the trained DNN
        """
        # Prepare features
        X = miner_pool[self.features]
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Flatten predictions to ensure we have a 1D array
        predictions_flat = predictions.flatten()
        
        # Select miner with highest efficiency score
        selected_index = np.argmax(predictions_flat)
        selected_miner = miner_pool.iloc[selected_index]
        score = predictions_flat[selected_index]
        
        return selected_miner, score
    
    def simulate_selection(self, miners_df, num_simulations=10):
        """
        Simulate multiple miner selections
        """
        results = []
        
        for i in range(num_simulations):
            # Create a random pool of 9 miners
            pool = miners_df.sample(9)
            selected_miner, score = self.select_miner(pool)
            
            results.append({
                'simulation': i+1,
                'selected_miner': selected_miner['miner_id'],
                'efficiency_score': score,
                'blocks_mined': selected_miner['blocks_mined'],
                'efficiency': selected_miner['efficiency'],
                'profitability': selected_miner['profitability']
            })
        
        return pd.DataFrame(results)
    
    def save_scaler(self, path):
        """Save the feature scaler"""
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")