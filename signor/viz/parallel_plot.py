import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/csv/iris.csv')
pd.plotting.parallel_coordinates(
        df, 'Name',
        color=('#556270', '#4ECDC4', '#C7F464'))
plt.show()