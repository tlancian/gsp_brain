import pandas as pd
import utils

data = pd.read_csv("dataset_rec.csv", index_col = False)



combos = [["gft", "dg", "bc", "cc"], ["gft", "dg", "bc"], ["gft", "dg", "cc"],
          ["gft", "bc", "cc"], ["gft"], ["dg"], ["bc"], ["cc"]]



for combo in combos:
        print(combo)
        print(utils.classificator(data, combo))
