# import libraries

import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn


def main(args):
    # enable autologging
    mlflow.autolog()

    # read data
    df = pd.read_parquet(Path(args.train_data) / "train.parquet" )

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(n_estimators = args.regressor__n_estimators,
                                  bootstrap = args.regressor__bootstrap,
                                  max_depth = args.regressor__max_depth,
                                  max_features = args.regressor__max_features,
                                  min_samples_leaf = args.regressor__min_samples_leaf,
                                  min_samples_split = args.regressor__min_samples_split,
                                  random_state=0)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)
    
    # train model
    model.fit(X_train, y_train)
    
    #mlflow.sklearn.save_model(model, args.model_output)
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")
    
    # Log the model using mlflow
    mlflow.sklearn.log_model(model, args.model_output)

    # Save the model using mlflow
    mlflow.sklearn.save_model(model, args.model_output)

# function that reads the data
def get_data(data_path):

    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--train_data", dest='train_data',
                        type=str)
    #parser.add_argument("--reg_rate", dest='reg_rate',
    #                    type=float, default=0.01)
    parser.add_argument("--model_output", dest='model_output',
                        type=str)
    # classifier specific arguments
    parser.add_argument('--regressor__n_estimators', type=int, default=500,
                        help='Number of trees')
    parser.add_argument('--regressor__bootstrap', type=int, default=1,
                        help='Method of selecting samples for training each tree')
    parser.add_argument('--regressor__max_depth', type=int, default=10,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--regressor__max_features', type=str, default='auto',
                        help='Number of features to consider at every split')
    parser.add_argument('--regressor__min_samples_leaf', type=int, default=4,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--regressor__min_samples_split', type=int, default=5,
                        help='Minimum number of samples required to split a node')

    args = parser.parse_args()                    
    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()
    
    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}"
    ]

    for line in lines:
        print(line)


    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
