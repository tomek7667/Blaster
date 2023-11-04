import json


def load_data(path="prepared/prepared_1697562094237.json"):
    with open(path, "r") as f:
        return json.load(f)
