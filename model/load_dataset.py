# import necessary modules
import os
import librosa

# function to load dataset from paths
def load(path):
    # prepare empty dictionary to store dataset (path, label, duration)
    dataset = {
        "paths": [],
        "labels": [],
        "durations": []
    }

    # walk through the path and append the data to the dictionary
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            dataset["paths"].append(path)

            label = filename.split("-")[0]
            dataset["labels"].append(label)

            duration = round(librosa.get_duration(path = path), 4)
            dataset["durations"].append(duration)
    
    # return the dictionary
    return dataset