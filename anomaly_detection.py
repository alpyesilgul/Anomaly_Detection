from sklearn.ensemble import IsolationForest
from utils import load_dataset


def model(dataset):
    print("Preparing dataset...")
    data = load_dataset(dataset, bins=(3, 3, 3))
    print("Fitting anomaly detection model...")
    model = IsolationForest(n_estimators=100, contamination=0.01,
                            random_state=42)
    model.fit(data)

    return model
