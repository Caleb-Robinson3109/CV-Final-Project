"""
model.py
This file will take in a dataset and output the extimated parameters.
The model that we will have in this file is a SVM, Random Forest, and Linear Regression.
At the minimum on of these models will be created. Others will be created time permitting.
"""
import torch
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load training and testing data
def load_dataset(data_path):
    data = torch.load(data_path, map_location="cpu")
    N = data['pred_theta'].shape[0]
    dataset = []

    for i in range(N):
        sample = np.concatenate([
            data['pred_theta'][i].reshape(-1).numpy(force=True),
            data['pred_beta'][i].reshape(-1).numpy(force=True),
            data['pred_cam'][i].reshape(-1).numpy(force=True),
            data['intrinsics'][i].reshape(-1).numpy(force=True),
            data['keypoints_2d'][i].reshape(-1).numpy(force=True),
            data['refined_thetas'][i].reshape(-1).numpy(force=True),
            data['refined_shape'][i].reshape(-1).numpy(force=True),
            data['refined_cam'][i].reshape(-1).numpy(force=True),
            data['refined_vertices'][i].reshape(-1).numpy(force=True),
            data['GT_pose'][i].reshape(-1).numpy(force=True),
            data['GT_beta'][i].reshape(-1).numpy(force=True),
        ])

        dataset.append(sample)

    del data
    return np.array(dataset)

parent_dir = os.path.dirname(os.getcwd())
train_data_path = os.path.join(parent_dir, "data", "kitro-refined", "train1.pt")
test_data_path = os.path.join(parent_dir, "data", "kitro-refined", "test.pt")
train_dataset = load_dataset(train_data_path)
test_dataset = load_dataset(test_data_path)
print("Loaded training and test data.")

GT_pose = 9
GT_beta = 10
y_columns = [("GT_pose", GT_pose), ("GT_beta", GT_beta)]
for name, col in y_columns:
    print(name, col)
    X_train = np.delete(train_dataset, col, axis=1)
    y_train = train_dataset[:, col]
    X_test = np.delete(test_dataset, col, axis=1)
    y_test = test_dataset[:, col]

    # Create a Random Forest Regressor
    # n_estimators: number of trees in the forest
    # random_state: for reproducibility
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_regressor.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    np.save(f"kitro_rf_{name}.npy", y_pred)
