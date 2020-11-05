from torchtext import data
import io
import os
import zipfile
import tarfile
import gzip
import shutil
from functools import partial

from .example import Example
from .utils import pandas_df_reader
from .utils import pandas_df_pickle_reader


class PandasDataset(data.Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
        """
        format = format.lower()
        make_example = {
            'dataframe': Example.fromDataFrame,
            'dataframe-pickled': Example.fromDataFramePickled}[format]

        # In the TabularDataset class, reader is a generator
        reader = None
        if format.lower() == 'dataframe':
            reader = pandas_df_reader(path, **csv_reader_params)
        elif format.lower() == 'dataframe-pickled':
            reader = pandas_df_pickle_reader(path, **csv_reader_params)


        # line in reader is like
        # ['000000000000000000000000000000000000000000000010000000000000000000000000000000000000100000', 'U.S. APPEARS TO TOLERATE FURTHER DLR DECLINE.
        # reader is a generator (using yield)
        examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(PandasDataset, self).__init__(examples, fields, **kwargs)
