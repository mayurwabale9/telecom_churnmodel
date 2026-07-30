# Data Manipulation Libraries
import numpy as np
import pandas as pd


def data_ingestion():
    df = pd.read_csv(r'C:\telecom_churnmodel\data\churn.csv')

    return df