import numpy as np
import pandas as pd
import argparse
import os

def npy_to_csv(npy_path, csv_path, feature_names=None):
    
    #feature_names=["% WEIGHTED ILI","%UNWEIGHTED ILI","AGE 0-4","AGE 5-24","ILITOTAL","NUM. OF PROVIDERS","OT"]
    
    # Load the .npy file
    data = np.load(npy_path)  # shape: [samples, pred_len, features]

    # Flatten it to 2D: each row is one timestep in the prediction
    num_samples, pred_len, num_features = data.shape
    data_2d = data.reshape(num_samples * pred_len, num_features)

    # Create DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(num_features)]

    df = pd.DataFrame(data_2d, columns=feature_names)

    # Save as CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True, help='Path to input .npy file')
    parser.add_argument('--csv', required=True, help='Path to output .csv file')
    args = parser.parse_args()

    npy_to_csv(args.npy, args.csv)
