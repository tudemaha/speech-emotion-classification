import os
import librosa

def load(path):
    dataset = {
        "paths": [],
        "labels": [],
        "durations": []
    }

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            dataset["paths"].append(path)

            label = filename.split("-")[0]
            dataset["labels"].append(label)

            duration = round(librosa.get_duration(path = path), 4)
            dataset["durations"].append(duration)
        
    return dataset