import warnings
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests




URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/airlines_data.xlsx"

filepath = '../data/airlines_data.xlsx'

if not os.path.isfile(filepath):
   pass