import pandas as pd

data = pd.read_csv('namegen/data/finalData.csv')

names = data[['name_en']]

names.to_csv('namegen/data/names.csv')