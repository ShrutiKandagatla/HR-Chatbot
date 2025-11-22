import pandas as pd

def load_faq(path='data/faqs.csv'):
    return pd.read_csv(path)