import numpy as np
import pandas as pd

import folium
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

def addPoint(scat, new_point, c='k'):
    old_off = scat.get_offsets()
    new_off = np.concatenate([old_off,np.array(new_point, ndmin=2)])
    old_c = scat.get_facecolors()
    new_c = np.concatenate([old_c, np.array(matplotlib.colors.to_rgba(c), ndmin=2)])

    scat.set_offsets(new_off)
    scat.set_facecolors(new_c)

    scat.axes.figure.canvas.draw_idle()
  
df = pd.read_csv('/home/ricardohb/Downloads/2000-16-traffic-flow-england-scotland-wales/accidents_2012_to_2014.csv', sep=',')

df2 = df[df['Accident_Severity'] == 1]

df_location = df2[['Longitude','Latitude','Number_of_Casualties']]
latitudes = [] 
longitudes = []
for index, row in df_location.iterrows():
    for i in range(int(row['Number_of_Casualties'])):
        latitudes.append(row['Latitude'])
        longitudes.append(row['Longitude'])

df2 = pd.DataFrame(list(zip(latitudes, longitudes)), columns = ['Latitude', 'Longitude'])
df_location = pd.DataFrame(list(zip(latitudes, longitudes)), columns = ['Latitude', 'Longitude'])
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df_location)
accident_location = DBSCAN(eps=0.01, min_samples=50).fit(df_scaled)
# accident_location = DBSCAN(eps=0.005, min_samples=20).fit(df_scaled)
pd.options.mode.chained_assignment = None
df2['labels'] = accident_location.labels_

# plt.scatter(df2['Longitude'], df2['Latitude'], c=df2['labels'])
# plt.show()
m = folium.Map([df2['Latitude'].iloc[0], df2['Longitude'].iloc[0]], zoom_start=5)

df2 = df2[df2['labels'] != -1]

colors = ['grey', 'purple', 'blue', 'green', 'orange', 'red']

for index, row in df2.iterrows():
    folium.Circle(
        radius=50,
        location=[row['Latitude'], row['Longitude']],
        color=colors[int(row['labels']) % len(colors)],
        fill=True,
    ).add_to(m)

m.save('index.html')
# What's the meaning of 0.01?? (i.e.: 25 or more accidents in a radious of x distance 0.01 determines the x)

# gmap = gmplot.GoogleMapPlotter(df2['Latitude'].iloc[0], df2['Longitude'].iloc[0], 5)
# gmap.scatter(df2['Latitude'], df2['Longitude'])

# gmap.draw('test1.html')