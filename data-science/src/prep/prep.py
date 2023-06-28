# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import os
import mlflow


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
   
    args = parser.parse_args()

    return args

def main(args):
    # read data
    df = get_data(args.raw_data)

    cleaned_data = clean_data(df)

    normalized_data = normalize_data(cleaned_data)

    # output_df = normalized_data.to_csv((Path(args.output_data) / "diabetes.csv"), index = False)
    # Split data into train, val and test datasets

    random_data = np.random.rand(len(normalized_data))

    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85

    train = normalized_data[msk_train]
    val = normalized_data[msk_val]
    test = normalized_data[msk_test]

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('val size', val.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    train.to_parquet((Path(args.train_data) / "train.parquet"))
    val.to_parquet((Path(args.val_data) / "val.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)

    # Count the rows and print the result
    row_count = (len(df))
    print('Preparing {} rows of data'.format(row_count))
    
    return df

# function that removes missing values
def clean_data(df):
    df = df.dropna()
    
    return df

# function that normalizes the data
def normalize_data(df):
    scaler = MinMaxScaler()
    num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()