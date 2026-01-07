import pandas as pd
from datetime import datetime

HISTORY_FILE = "data/history.csv"

def save_history(data):
    data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    try:
        old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([old, df], ignore_index=True)
    except:
        pass
    df.to_csv(HISTORY_FILE, index=False)

def load_history():
    try:
        return pd.read_csv(HISTORY_FILE)
    except:
        return None
