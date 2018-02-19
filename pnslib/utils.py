"""Utility functions.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import os
try:
    import subprocess32 as sp
except ImportError:
    import subprocess as sp

import gzip
import numpy as np

import pnslib


def speak(msg_str, options=None):
    """Text-to-Speech via espeak and mplayer.

    # Parameters
    msg_str : str
        The candidate string to speak
    options : list
        espeak program options, use default config if None
        default config
        ["-ven+f3", "-k5", "-s150"]

    # Returns
    cmd_out : str
        command output as string
    """
    speak_base = ["/usr/bin/espeak"]
    if options is not None and isinstance(options, list):
        speak_config = options
    else:
        speak_config = ["-ven+f3", "-k5", "-s150"]
    mplayer_base = ["/usr/bin/mplayer", "/tmp/pns_speak.wav"]

    speak_cmd = speak_base+speak_config + \
        [msg_str, ">", "/tmp/pns_speak.wav", "|"]+mplayer_base

    return sp.check_output(speak_cmd, stderr=sp.PIPE).decode("UTF-8")


def fashion_mnist_download(train=True, test=True, labels=True):
    """Download Fashion MNIST.

    The original resource of Fashion MNIST is here:
    https://github.com/zalandoresearch/fashion-mnist

    The function uses wget to download

    # Parameters
    train : bool
        if download training dataset
    test : bool
        if download testing dataset
    labels : bool
        if download labels

    # Returns
    downloaded datasets in ~/.pnslibres
    """
    # define datasets URL
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    train_x_name = "train-images-idx3-ubyte.gz"
    train_y_name = "train-labels-idx1-ubyte.gz"
    test_x_name = "t10k-images-idx3-ubyte.gz"
    test_y_name = "t10k-labels-idx1-ubyte.gz"

    # construct save path
    save_path = os.path.join(pnslib.PNSLIB_DATA, "fashion-mnist")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if train is True:
        train_x_path = os.path.join(save_path, train_x_name)
        sp.check_output(["wget", base_url+train_x_name,
                         "-O", train_x_path],
                        stderr=sp.PIPE).decode("UTF-8")
        if labels is True:
            train_y_path = os.path.join(save_path, train_y_name)
            sp.check_output(["wget", base_url+train_y_name,
                             "-O", train_y_path],
                            stderr=sp.PIPE).decode("UTF-8")

    if test is True:
        test_x_path = os.path.join(save_path, test_x_name)
        sp.check_output(["wget", base_url+test_x_name,
                         "-O", test_x_path],
                        stderr=sp.PIPE).decode("UTF-8")
        if labels is True:
            test_y_path = os.path.join(save_path, test_y_name)
            sp.check_output(["wget", base_url+test_y_name,
                             "-O", test_y_path],
                            stderr=sp.PIPE).decode("UTF-8")


def fashion_mnist_load(data_type="full"):
    """Loading Fashion MNIST dataset.

    # Parameters
    data_type : str
        full : training, testing and labels
        train : training and its labels
        test : testing and its labels
        train-only : only training dataset
        test-only : only testing dataset

    # Returns
    dataset : tuple
        a tuple of requested data
        full: (train_x, train_y, test_x, test_y)
        train: (train_x, train_y)
        test: (test_x, test_y)
        train_only: (train)
        test_only: (test)
    """
    save_path = os.path.join(pnslib.PNSLIB_DATA, "fashion-mnist")
    if not os.path.isdir(save_path):
        print ("Downloading dataset.")
        fashion_mnist_download(train=True, test=True, labels=True)

    train_x_name = "train-images-idx3-ubyte.gz"
    train_y_name = "train-labels-idx1-ubyte.gz"
    test_x_name = "t10k-images-idx3-ubyte.gz"
    test_y_name = "t10k-labels-idx1-ubyte.gz"

    dataset = ()
    if data_type in ["train", "train_only", "full"]:
        train_x_path = os.path.join(save_path, train_x_name)
        with gzip.open(train_x_path, "rb") as x_file:
            train_x = np.frombuffer(x_file.read(), dtype=np.uint8,
                                    offset=16).reshape(60000, 28, 28, 1)
            x_file.close()

        dataset = dataset+(train_x,) if train_x is not None else dataset
        if data_type != "train_only":
            train_y_path = os.path.join(save_path, train_y_name)
            with gzip.open(train_y_path, "rb") as y_file:
                train_y = np.frombuffer(y_file.read(), dtype=np.uint8,
                                        offset=8)
                y_file.close()
            dataset = dataset+(train_y,) if train_y is not None else dataset

    if data_type in ["test", "test_only", "full"]:
        test_x_path = os.path.join(save_path, test_x_name)
        with gzip.open(test_x_path, "rb") as x_file:
            test_x = np.frombuffer(x_file.read(), dtype=np.uint8,
                                   offset=16).reshape(10000, 28, 28, 1)
            x_file.close()
        dataset = dataset+(test_x,) if test_x is not None else dataset
        if data_type != "test_only":
            test_y_path = os.path.join(save_path, test_y_name)
            with gzip.open(test_y_path, "rb") as y_file:
                test_y = np.frombuffer(y_file.read(), dtype=np.uint8,
                                       offset=8)
                y_file.close()
            dataset = dataset+(test_y,) if test_y is not None else dataset

    return dataset
