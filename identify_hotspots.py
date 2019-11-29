import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import preprocessing
  
df = pd.read_csv('/home/ricardohb/Downloads/2000-16-traffic-flow-england-scotland-wales/accidents_2012_to_2014.csv', sep=',')

df2 = df[df['Accident_Severity'] == 1]
df_location = df2[['Location_Easting_OSGR','Location_Northing_OSGR']]
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df_location)
accident_location = DBSCAN(eps=0.01, min_samples=25).fit(df_scaled)
# What's the meaning of 0.01?? (i.e.: 25 or more accidents in a radious of x distance 0.01 determines the x)

plt.scatter(df2['Location_Easting_OSGR'], df2['Location_Northing_OSGR'], c=accident_location.labels_, cmap = 'jet')
plt.show()