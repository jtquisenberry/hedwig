import pandas as pd
df = pd.read_csv(r'E:\Development\corpora\hedwig-data\datasets\Reuters\train.tsv', header=None, sep='\t', names=['labels', 'text'])
print(df.head())
df.to_pickle(r'E:\Development\corpora\hedwig-data\datasets\Reuters_PD\reuters_pd.pkl')
a = 1