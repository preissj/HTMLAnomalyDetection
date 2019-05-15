from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import re

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.max_columns = 16
pd.options.display.float_format = '{:.1f}'.format

rx_dict = {
    'first_line': re.compile(r"(?P<request_type>GET|HEAD|POST|PUT|DELETE|CONNECT|OPTIONS|TRACE) (?P<URI>.*)\n"),
    'user_agent': re.compile(r"User-Agent: (?P<user_agent>.*)"),
    'pragma': re.compile(r"Pragma: (?P<pragma>.*)"),
    'cache_control': re.compile(r"Cache-control: (?P<cache_control>.*)"),
    'accept': re.compile(r"Accept: (?P<accept>.*)"),
    'accept_encoding': re.compile(r"Accept-Encoding: (?P<accept_encoding>.*)"),
    'accept_charset': re.compile(r"Accept-Charset: (?P<accept_charset>.*)"),
    'accept_language': re.compile(r"Accept-Language: (?P<accept_language>.*)"),
    'host': re.compile(r"Host: (?P<host>.*)"),
    'cookie': re.compile(r"Cookie: (?P<cookie>.*)"),
    'content_type': re.compile(r"Content-Type: (?P<content_type>.*)"),
    'connection': re.compile(r"Connection: (?P<connection>.*)"),
    'content_length': re.compile(r"Content-Length: (?P<content_length>.*)")
}


def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match is not None:
            return key, match

    return None, None


def parse_file(filepath):
    """
    Parse text at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file_object to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        is_first_line = True
        record = {
            'request_type': "",
            'URI': "",
            'user_agent': "",
            'pragma': "",
            'cache_control': "",
            'accept': "",
            'accept_encoding': "",
            'accept_charset': "",
            'accept_language': "",
            'host': "",
            'cookie': "",
            'content_type': "",
            'connection': "",
            'content_length': "",
            'parameters': ""
        }

        while line:
            # at each line check for a match with a regex
            key, match = _parse_line(line)
            if key is None:
                if line != "\n":
                    record['parameters'] = line.strip()
            elif key == "first_line":
                if not is_first_line:
                    data.append(record)
                record = {
                    'request_type': match.group("request_type"),
                    'URI': match.group("URI"),
                    'user_agent': "",
                    'pragma': "",
                    'cache_control': "",
                    'accept': "",
                    'accept_encoding': "",
                    'accept_charset': "",
                    'accept_language': "",
                    'host': "",
                    'cookie': "",
                    'content_type': "",
                    'connection': "",
                    'content_length': "",
                    'parameters': ""
                }
                is_first_line = False
            else:
                record[key] = match.group(key)

            # append the dictionary to the data list
            # data.append(record)

            line = file_object.readline()
        data.append(record)
    # create a pandas DataFrame from the list of dicts
    labels = ['request_type', 'URI', 'user_agent', 'pragma', 'cache_control', 'accept', 'accept_encoding', 'accept_charset', 'accept_language', 'host', 'cookie', 'content_type', 'connection', 'content_length', 'parameters']
    data = pd.DataFrame.from_records(data, columns=labels)
    data = pd.DataFrame(data)
    return data


normalTraffic = parse_file("normalTrafficTraining.txt")
anomalousTraffic = parse_file("anomalousTrafficTest.txt")