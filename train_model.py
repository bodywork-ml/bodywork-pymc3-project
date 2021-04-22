"""
This module generates synthetic one-dimensional regression data and
trains a sklearn.dummy.DummyRegressor using the mean strategy. The model
is persisted locally as 'dummy_model.joblib' andn the model metrics are
persisted locally as 'dummy_model_metrics.txt'.
"""
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# create dummy regression data
n_observations = 1000
np.random.seed(42)
X = np.random.randn(n_observations)
y = 0.42 * X + np.sqrt(1 - 0.42 * 0.42) * np.random.randn(n_observations)

# train dummy model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, y_train)

# compute dummy model metrics
mse = mean_squared_error(y_test, dummy_model.predict(X_test))

# persist dummy model and metrics
joblib.dump(dummy_model, 'dummy_model.joblib')
with open('dummy_model_metrics.txt', 'w') as f:
    f.write(f'mean_squared_error: {mse}\n')
