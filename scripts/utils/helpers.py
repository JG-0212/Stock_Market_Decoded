import yaml
from datetime import datetime

def read_yaml(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

def get_jan_first_years_ago(years_back):
    today = datetime.today()
    return datetime(today.year - years_back, 1, 1)

def data_statistics(data):
    print('Mean:', data.mean(axis=0))
    print('Max', data.max())
    print('Min', data.min())
    print('Std dev:', data.std(axis=0))