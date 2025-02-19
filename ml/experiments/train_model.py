# ml/experiments/train_model.py
from comet_ml import Experiment
import pickle
from ml.models.model import build_model
from ml.data.load_data import load_training_data

# Initialize comet_ml experiment (replace with your actual API key and project details)
experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="ml-web-app",
    workspace="your_workspace",
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
