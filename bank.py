import pandas as pd
import numpy as np

bank = pd.read_csv("bank-full.csv", delimiter = ";")

#bank.drop(columns = ["contact"], inplace = True)
#bank = bank[~bank.isin(["unknown"]).any(axis = 1)]

for index in range(len(bank)):
    if bank.iloc[index, 13] != -1 and bank.iloc[index, 15] == "unknown":
        print(1)
        print(index)
    if bank.iloc[index, 13] == -1 and bank.iloc[index, 15] != "unknown":
        #doesn't print anything here
        print(2)
        print(index)