import pandas as pd, os

LOG_FILE = "monitoring/current.csv"
os.makedirs("monitoring", exist_ok=True)

def append_input(record: dict):
    df = pd.DataFrame([record])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
