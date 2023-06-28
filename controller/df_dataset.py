import pandas as pd
from model.load_dataset import load

def dataset_df(path):
    dataset_dict = load(path)
    df = pd.DataFrame(dataset_dict)
    df.columns = ["Path", "Label", "Duration"]
    return df