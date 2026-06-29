import numpy as np
import pandas as pd
import os

def generate_data(n_records=810_909, seed=42):
    """
    Generate synthetic Bitcoin-like block records.
    Returns a DataFrame with n_records rows.
    """
    rng = np.random.default_rng(seed)
    block_ids = np.arange(n_records)
    # number of transactions in a block
    n_transactions = rng.integers(1, 6_936, size=n_records)
    # size of a block in bytes (max of 2MB)
    size = rng.integers(215, 2_000_001, size=n_records)
    # difficulty of the hash required for PoW (exponentially increasing)
    difficulty = np.linspace(1, 1e13, n_records)
    # amount of BTC moved in a block (eg: 50 transactions of 2BTC = 100 transaction_volume)
    # 0 BTC to 21 million BTC (MAX amount to exist)
    transaction_volume = rng.uniform(0, 2_100_000, size=n_records)
    
    df = pd.DataFrame({
        'block_id': block_ids,
        'n_transactions': n_transactions,
        'size': size,
        'difficulty': difficulty,
        'transaction_volume': transaction_volume
    })
    
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_data()
    df.to_csv('data/bitcoin_blocks.csv', index=False)
    print(f"Generated {len(df)} rows, saved to data/bitcoin_blocks.csv")