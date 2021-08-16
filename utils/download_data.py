import json
import numpy as np
# import pandas as pd
from urllib.parse import unquote
import os
import urllib
import urllib.request
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed


def down_image(url):
    print(url)
    if os.path.exists('../data/train_stage2/' + url.split('/')[-1]):
        return
    urllib.request.urlretrieve(url, '../data/train_stage2/' + url.split('/')[-1])


if __name__ == '__main__':
    train = [
        # "Xeon1OCR_round1_train1_20210526.csv",
        # "Xeon1OCR_round1_train_20210524.csv",
        # "Xeon1OCR_round1_train2_20210526.csv"
        pd.read_csv("../data/OCR复赛数据集01.csv"),
        pd.read_csv("../data/OCR复赛数据集02.csv")
    ]
    # df = []

    urls = []
    # train data
    df = pd.concat(train)
    for row in df.iterrows():
        path = json.loads(row[1]['原始数据'])['tfspath']
        urls.append(path)

    Parallel(n_jobs=-1)(delayed(down_image)(url) for url in tqdm(urls))

    # for csv in train:
    #     df.append(pd.read_csv('../data/' + csv))

    # df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
    # df["链接"].to_csv("stage2_train.txt", header=False, index=False)

    # test data
    # test_df = []
    # for i, csv in enumerate(test):
    #     df = pd.read_csv('../data/' + csv)
    #     test_df.append(df)
    #     df["链接"] = df["原始数据"].apply(lambda x: json.loads(x)["tfspath"])
    #     df["链接"].to_csv(f"test{i + 1}.txt", header=False, index=False)
