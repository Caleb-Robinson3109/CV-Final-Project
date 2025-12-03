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
    dataset_X = []
    dataset_y = []

    for i in range(N):
        sample_X = np.concatenate([
            data['pred_theta'][i].reshape(-1).numpy(force=True),
            data['pred_beta'][i].reshape(-1).numpy(force=True),
            data['pred_cam'][i].reshape(-1).numpy(force=True),
            data['intrinsics'][i].reshape(-1).numpy(force=True),
            data['keypoints_2d'][i].reshape(-1).numpy(force=True),
            data['refined_vertices'][i].reshape(-1).numpy(force=True),
            data['GT_pose'][i].reshape(-1).numpy(force=True),
            data['GT_beta'][i].reshape(-1).numpy(force=True),
        ])
        sample_y = np.concatenate([
            data['refined_thetas'][i].reshape(-1).numpy(force=True),
            data['refined_shape'][i].reshape(-1).numpy(force=True),
            data['refined_cam'][i].reshape(-1).numpy(force=True),
        ])

        dataset_X.append(sample_X)
        dataset_y.append(sample_y)

    del data
    dataset_X = np.array(dataset_X)
    dataset_y = np.array(dataset_y)
    print(dataset_X.shape, dataset_y.shape)
    return dataset_X, dataset_y

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_path = os.path.join(parent_dir, "data", "kitro-refined", "refined061.pt")
test_data_path = os.path.join(parent_dir, "data", "kitro-refined", "refined065.pt")
X_train, y_train = load_dataset(train_data_path)
X_test, y_test = load_dataset(test_data_path)
print("Loaded training and test data.")

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
np.save(os.path.join(parent_dir, "data", "random-forest", "kitro_rf.npy"), y_pred)
