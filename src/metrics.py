def load_metrics():
    metrics = {}
    try:
        with open("metrics.txt", "r") as f:
            for line in f.readlines():
                key, value = line.strip().split(":")
                metrics[key.strip()] = float(value.strip().replace("%", ""))
    except:
        pass
    return metrics
