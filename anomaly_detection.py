from sklearn.ensemble import IsolationForest
from utils import load_dataset


def model(dataset):
    print("[INFO] preparing dataset...")
    data = load_dataset(dataset, bins=(3, 3, 3))
    print("[INFO] fitting anomaly detection model...")
    model = IsolationForest(n_estimators=100, contamination=0.01,
                            random_state=42)
    model.fit(data)

    return model
