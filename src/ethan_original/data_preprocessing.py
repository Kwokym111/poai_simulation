import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess blockchain data"""
        # Check if file is CSV or Excel
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:  # Excel file
            df = pd.read_excel(self.data_path)
        
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if we have enough data
        if len(df) < 100:
            print("Warning: Dataset is very small. Results may not be reliable.")
        
        # Rename columns to match expected names
        column_mapping = {
            'height': 'block_id',
            'tx_count': 'n_transactions',
            'output_amount': 'transaction_volume'
        }
        
        # Apply mapping for columns that exist in the dataset
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Convert transaction_volume from satoshis to BTC (if present)
        if 'transaction_volume' in df.columns:
            df['transaction_volume'] = df['transaction_volume'] / 1e8
        
        # Check if required columns exist
        required_columns = ['block_id', 'timestamp', 'n_transactions', 'transaction_volume', 'difficulty']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def create_miner_simulation(self, df):
        """Simulate miners from blockchain data with improved characteristics"""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by block_id
        df = df.sort_values('block_id')
        
        # Create more diverse miners (increased from 50 to 100 for better training)
        num_miners = 100
        df['miner_id'] = df.index % num_miners
        
        # Group by miner_id
        miner_groups = df.groupby('miner_id')
        
        miner_data = []
        for miner_id, group in miner_groups:
            # Calculate additional features for better diversity
            avg_fee = (group['n_transactions'] * group['transaction_volume']).mean()
            fee_volatility = (group['n_transactions'] * group['transaction_volume']).std()
            
            miner_data.append({
                'miner_id': f"miner_{miner_id}",
                'blocks_mined': len(group),
                'avg_transactions': group['n_transactions'].mean(),
                'avg_volume': group['transaction_volume'].mean(),
                'avg_fee': avg_fee,
                'fee_volatility': fee_volatility if not np.isnan(fee_volatility) else 0,
                'last_mined': group['timestamp'].max(),
                'difficulty': group['difficulty'].mean(),
                'total_fees': (group['n_transactions'] * group['transaction_volume']).sum(),
                'avg_block_size': group['size'].mean()
            })
        
        miners_df = pd.DataFrame(miner_data)
        
        # Calculate age (time since last mined)
        miners_df = miners_df.sort_values('last_mined')
        miners_df['age'] = miners_df['last_mined'].diff().dt.total_seconds().fillna(0)
        
        # Calculate effective value with improved formula
        miners_df['efficiency'] = miners_df['avg_transactions'] / (miners_df['avg_volume'] + 1)
        miners_df['profitability'] = miners_df['total_fees'] / (miners_df['blocks_mined'] + 1)
        
        return miners_df
    
    def prepare_training_data(self, miners_df):
        """Prepare data for DNN training with improved features"""
        features = [
            'blocks_mined', 'avg_transactions', 'avg_volume', 'difficulty', 
            'avg_fee', 'fee_volatility', 'avg_block_size', 'age', 'profitability'
        ]
        target = 'efficiency'  # Use efficiency as target
        
        X = miners_df[features]
        y = miners_df[target]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create binary classification (efficient vs inefficient)
        y_binary = (y > y.median()).astype(int)
        
        # Handle small datasets
        if len(X_scaled) < 10:
            raise ValueError(f"Not enough data for training. Only {len(X_scaled)} samples available.")
        
        # Use stratified split for better balance
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"Features used: {features}")
        print(f"Feature statistics:")
        print(X.describe())
        
        return X_train, X_test, y_train, y_test, features