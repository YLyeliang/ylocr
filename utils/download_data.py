import json
import numpy as np
import pandas as pd
from urllib.parse import unquote

if __name__ == '__main__':
    train = [
        "Xeon1OCR_round1_train1_20210526.csv",
        "Xeon1OCR_round1_train_20210524.csv",
        "Xeon1OCR_round1_train2_20210526.csv"
    ]
    test = [
        "Xeon1OCR_round1_test1_20210528.csv",
        "Xeon1OCR_round1_test2_20210528.csv",
        "Xeon1OCR_round1_test3_20210528.csv"
    ]
    df = []

    # train data
    for csv in train:
        df.append(pd.read_csv('../data/' + csv))
    df = pd.concat(df)
    df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
    df["链接"].to_csv("train.txt", header=False, index=False)

    # test data
    test_df = []
    for i, csv in enumerate(test):
        df = pd.read_csv('../data/' + csv)
        test_df.append(df)
        df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
        df["链接"].to_csv(f"test{i + 1}.txt", header=False, index=False)
