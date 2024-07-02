import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Create a dummy dataset
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Train a dummy model
model = RandomForestClassifier()
model.fit(X, y)

# Save the dummy model to 'model.pkl'
joblib.dump(model, 'model.pkl')
