import six
import json
from functools import reduce


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """
    @classmethod
    def fromDataFrame(cls, data, fields, field_to_index=None):
        if field_to_index is None:
            return cls.fromlist(data, fields)
        else:
            assert(isinstance(fields, dict))
            data_dict = {f: data[idx] for f, idx in field_to_index.items()}
            return cls.fromdict(data_dict, fields)

    @classmethod
    def fromDataFramePickled(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                # Handle field tuples
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    setattr(ex, name, field.preprocess(val))
        return ex


