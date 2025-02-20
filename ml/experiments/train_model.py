# ml/experiments/train_model.py
from comet_ml import Experiment, start
import pickle
from ml.models.model import build_model
from ml.data.load_data import load_training_data

from ml.keys import API_KEY, PROJECT_NAME, WORKSPACE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For splitting the data and evaluating the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# For the regression model
from sklearn.ensemble import GradientBoostingRegressor

# For saving the trained model
import joblib

data_path = "uber.csv"  
df = pd.read_csv(data_path)

df.head()

# Initialize comet_ml experiment (replace with your actual API key and project details)
experiment = start(
  api_key=API_KEY,
  project_name=PROJECT_NAME,
  workspace=WORKSPACE
)

# Log hyperparameters
params = {"learning_rate": 0.01, "epochs": 50}
experiment.log_parameters(params)

# Load data and build model
X_train, y_train = load_training_data()
model = build_model(params)

# Train model (example training loop)
for epoch in range(params["epochs"]):
    # ... training logic ...
    loss = 0.01 * (params["epochs"] - epoch)  # dummy loss calculation
    experiment.log_metric("loss", loss, step=epoch)

# Save the trained model
with open("ml/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

experiment.end()
