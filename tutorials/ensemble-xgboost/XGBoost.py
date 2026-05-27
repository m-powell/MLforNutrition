# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:51:14 2024

@author: dylan.hyde
"""

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = np.loadtxt('NHANES.csv',dtype=str, delimiter=",")