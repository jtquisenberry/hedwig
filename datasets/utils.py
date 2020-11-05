import six
import requests
import csv
from tqdm import tqdm
import os
import tarfile
import logging
import re
import sys
import zipfile
import pandas as pd


def pandas_df_pickle_reader(pickle_filename, num_top_rows=5, **kwargs):
    r"""Reads a Pandas DataFrame stored as a pickle.

    Arguments:
        pickle_filename: Fully-qualified path and filename to a pickled dataframe.

    Examples:
      #  >>> from .utils import pandas_df_pickle_reader
      #  >>>
      #  >>>
      #  >>>

    """

    df = pd.read_pickle(pickle_filename)
    df = df.head(num_top_rows)
    for index, row in df.iterrows():
        # print(row['labels'], row['text'])
        yield row['labels'], row['text']


def pandas_df_reader(pandas_dataframe, num_top_rows=5, **kwargs):
    r"""Reads a Pandas DataFrame and iterates over its rows using a generator.

        Arguments:
            pandas_dataframe: unicode csv data (see example below)

        Examples:
          #  >>> from .utils import pandas_df_reader
          #  >>>
          #  >>>
          #  >>>

        """

    df = pandas_dataframe
    df = df.head(num_top_rows)
    for index, row in df.iterrows():
        # print(row['labels'], row['text'])
        yield row['labels'], row['text']


if __name__ == '__main__':
    file = r"E:\Development\corpora\hedwig-data\datasets\Reuters_PD\reuters_pd.pkl"
    g = pandas_df_pickle_reader(file)
    for x in g:
        print(x[0])
