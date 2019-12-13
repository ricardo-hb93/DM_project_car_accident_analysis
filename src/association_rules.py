# -*- coding: utf-8 -*-
"""Association rules

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-z3ZyPK4Y3yy_lmQuq8R2KWU_QL7E7QQ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Libraries for importing datasets from Google Drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Configuring Google Drive file loading (run it only once)
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

link = 'https://drive.google.com/open?id=1MURORv9iCRNNZtZORoNlv3WtDAYcGYqJ'
id = link.split('=')[1]

downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile(id)  
df = pd.read_csv(id, sep=',')

df2 = df.loc[df['Accident_Severity'] == 2].sample(n=2500, random_state=1)
#print(df2.shape)
df2 = pd.concat([
    df2,df.loc[df['Accident_Severity'] == 3].sample(n=2500, random_state=1)
],ignore_index=True)
#print(df2.shape)

df2 = pd.concat([
    df2,df.loc[df['Accident_Severity'] == 1]
],ignore_index=True)
#print(df2.shape)

"""Downsampling"""

import random
import time

def create_dataset(dataset):
	random.seed(time.time())
	tmp = dataset.loc[dataset['Accident_Severity'] == 2].sample(n=2500, random_state = random.randrange(0,1000))
	
	tmp = pd.concat([
		tmp,dataset.loc[dataset['Accident_Severity'] == 3].sample(n=2500, random_state = random.randrange(0,1000))
	],ignore_index=True)
	
	
	tmp = pd.concat([
		tmp,dataset.loc[dataset['Accident_Severity'] == 1]
	],ignore_index=True)
	print(tmp.shape)
	return tmp

"""Preprocessing functions"""

#remove Unknown for Road_Type, Weather_Conditions
#remove nan for Road_Surface_Conditions
#remove None for Special_Conditions_at_Site, Carriageway_Hazards

def remove_nones(l):
	length1 = len(l)
	j=0
	while(j < length1):
		i = 0
		length = len(l[j])  #list length
		#print(row) 
		while(i<length):
			if(l[j][i]=='None' or l[j][i]=='Unknown' or l[j][i] is np.nan or l[j][i]=='nan'):
				l[j].remove (l[j][i])
				length = length -1  
				continue
			i = i+1
		j = j+1

def select_features(l):
	l['Accident_Severity'] = l['Accident_Severity'].map({1: 'Fatal', 2: 'Serious', 3: 'Slight'})
	l['Day_of_Week'] = l['Day_of_Week'].map({1: 'Sun', 2: 'Mon', 3: 'Tus', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'})
	l['Speed_limit'] = l['Speed_limit'].map({20: 'speed_20', 30: 'speed_30', 40: 'speed_40', 50: 'speed_50', 60: 'speed_60', 70: 'speed_70'})
	l['Urban_or_Rural_Area'] = l['Urban_or_Rural_Area'].map({1: 'Urban', 2: 'Rural'})	
	#Discretize time
	hours = pd.to_datetime(l['Time'], format='%H:%M').dt.hour
	l['Time'] = pd.cut(hours, 
                    bins=[0,6,12,18,24], 
                    include_lowest=True, 
                    labels=['Midnight','Morning','Evening','Night'])
	l['month'] = l['month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
	return l

def preprocess_dateset_for_assossiation_rules(dataset):
	# removing useless columns (more than 90% of the data have the same value)
	dataset.drop('Special_Conditions_at_Site', axis = 1,inplace = True)
	dataset.drop('Carriageway_Hazards',axis = 1,inplace = True)
	dataset.drop('Pedestrian_Crossing-Human_Control',axis = 1, inplace= True)
	# removing duplicate of coordinate system
	dataset.drop('Location_Northing_OSGR',axis = 1, inplace = True)
	dataset.drop('Location_Easting_OSGR',axis = 1,inplace = True)
	dataset.drop('Police_Force',axis = 1,inplace = True)
	dataset.drop('Number_of_Vehicles',axis = 1,inplace = True)
	dataset.drop('Number_of_Casualties',axis = 1,inplace = True)
	dataset.drop('Did_Police_Officer_Attend_Scene_of_Accident',axis = 1,inplace = True)
	dataset.drop('Accident_Index',axis = 1,inplace = True)
	dataset['month'] = pd.DatetimeIndex(dataset['Date']).month
	dataset.drop('Date',axis = 1,inplace = True)
	dataset.drop('Junction_Detail',axis = 1,inplace = True)
 
	#to be used later
	dataset.drop('Longitude',axis = 1,inplace = True)
	dataset.drop('Latitude',axis = 1,inplace = True)
	dataset.drop('1st_Road_Number',axis = 1,inplace = True)
	dataset.drop('2nd_Road_Number',axis = 1,inplace = True)
	dataset.drop('Year',axis = 1,inplace = True)
	
	dataset['Local_Authority_(District)'] = 'local_auth_' + df['Local_Authority_(District)'].astype(str)
	dataset['1st_Road_Class'] = '1st_road_class_' + df['1st_Road_Class'].astype(str)
	dataset['2nd_Road_Class'] = '2nd_road_class_' + df['2nd_Road_Class'].astype(str)
	return select_features(dataset)

def preprocess_dateset_for_assossiation_rules2(dataset):
  # for every column if it contains just one value (without add Nan to the count) drop it
  for col in dataset:
    if not dataset[col].value_counts().to_list():
      dataset.drop(col, axis = 1,inplace = True)

  # removig the index
  dataset.drop('Accident_Index',axis = 1, inplace = True)
  # removing useless columns (more than 90% of the data have the same value)
  dataset.drop('Special_Conditions_at_Site', axis = 1,inplace = True)
  dataset.drop('Carriageway_Hazards',axis = 1,inplace = True)
  dataset.drop('Pedestrian_Crossing-Human_Control',axis = 1, inplace= True)
  # removing duplicate of coordinate system
  dataset.drop('Location_Northing_OSGR',axis = 1, inplace = True)
  dataset.drop('Location_Easting_OSGR',axis = 1,inplace = True)
  dataset.drop('LSOA_of_Accident_Location',axis = 1,inplace = True)
  dataset.drop('1st_Road_Number',axis = 1, inplace = True)
  dataset.drop('2nd_Road_Number',axis = 1, inplace = True)
  dataset.drop('Police_Force',axis = 1, inplace = True)
  dataset.drop('Year',axis = 1, inplace = True)

  # removing junctioncontrol because it has 40% of nan values
  dataset.drop('Junction_Control',axis = 1,inplace = True)
  # REMOVE ALSO THE FIRST AND SECOND ORDER STUFF BECAUSE THEY ARE USEFUL ONLY FOR JUNCTIONS

  # removing the date because we just use the day of the week
  dataset['month'] = pd.DatetimeIndex(dataset['Date']).month
  dataset.drop('Date',inplace = True, axis = 1)

  # removing 13 nan rows in time, no way to impute it
  dataset.dropna(subset = ['Time'],inplace = True)
  dataset.dropna(subset = ['Did_Police_Officer_Attend_Scene_of_Accident'],inplace = True)

  # merging attribute values
  dataset.loc[dataset['Number_of_Vehicles'] > 2,'Number_of_Vehicles'] = '3+'
  dataset.loc[dataset['Number_of_Vehicles'] == 2,'Number_of_Vehicles'] = 'two'
  dataset.loc[dataset['Number_of_Vehicles'] == 1,'Number_of_Vehicles'] = 'one'
  dataset.loc[(dataset['Road_Type'] != 'Single carriageway') & (dataset['Road_Type'] != 'Dual carriageway'),'Road_Type'] = 'Other'
  dataset.loc[dataset['Pedestrian_Crossing-Physical_Facilities'] != 'No physical crossing within 50 meters', 'Pedestrian_Crossing-Physical_Facilities'] = 'Physical Crossing present'
  dataset.loc[(dataset['Weather_Conditions'] != 'Fine without high winds') & (dataset['Weather_Conditions'] != 'Raining without high winds'), 'Weather_Conditions'] = 'Other'
  dataset.loc[(dataset['Road_Surface_Conditions'] != 'Dry') & (dataset['Road_Surface_Conditions'] != 'Wet/Damp'),'Road_Surface_Conditions'] = 'Extreme_condition' 
  dataset.loc[(dataset['Number_of_Casualties'] != 1) & (dataset['Number_of_Casualties'] != 2),'Number_of_Casualties'] = '3+'
  dataset.loc[dataset['Number_of_Casualties'] == 2,'Number_of_Casualties'] = 'two'
  dataset.loc[dataset['Number_of_Casualties'] == 1,'Number_of_Casualties'] = 'one' 
  dataset.loc[(dataset['Light_Conditions'] == 'Daylight: Street light present') | (dataset['Light_Conditions'] == 'Darkness: Street lights present and lit'),'Light_Conditions'] = 'Proper lightning'
  dataset.loc[(dataset['Light_Conditions'] == 'Darkness: Street lights present but unlit') | (dataset['Light_Conditions'] == 'Darkeness: No street lighting'),'Light_Conditions'] = 'Insufficient lightning'
  dataset.loc[dataset['Light_Conditions'] == 'Darkness: Street lighting unknown','Light_Conditions'] = 'Unknown'
  dataset.loc[dataset['Speed_limit'] == 10,'Speed_limit'] = 20

  dataset['Local_Authority_(District)'] = 'local_auth_' + df['Local_Authority_(District)'].astype(str)
  dataset['1st_Road_Class'] = '1st_road_class_' + df['1st_Road_Class'].astype(str)
  dataset['2nd_Road_Class'] = '2nd_road_class_' + df['2nd_Road_Class'].astype(str)
  dataset['Number_of_Casualties'] = 'Number_of_Casualties = ' + df['Number_of_Casualties'].astype(str)
  dataset['Number_of_Vehicles'] = 'Number_of_Vehicles = ' + df['Number_of_Vehicles'].astype(str)
  dataset['Did_Police_Officer_Attend_Scene_of_Accident'] = 'Did_Police_Officer_Attend_Scene_of_Accident = ' + df['Did_Police_Officer_Attend_Scene_of_Accident'].astype(str)
  return select_features(dataset)

"""MAIN PROGRAM"""

!pip install pyfpgrowth
!pip install pygeohash

import pyfpgrowth
import numpy as np
import pygeohash as gh

#Transform Latitude and Longitude using geohash
df['geohash']=df.apply(lambda x: gh.encode(x.Latitude, x.Longitude, precision=3), axis=1)

rules_list = []
for iters in range(50):
	print("iteration "+str(iters)+" started...")
	df2 = create_dataset(df)
	df3 = preprocess_dateset_for_assossiation_rules(df2)
	df3 = df3.applymap(str)
	
	df3_list = df3.values.tolist()
	remove_nones(df3_list)	
	
	patterns = pyfpgrowth.find_frequent_patterns(df3_list, 700)
	rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
	
	rules_list.append(rules)

"""Create an dictionary that contains the association rules and their frequency"""

rules_dict = {}
for rules in rules_list:
	print("next_run......................")
	for i in rules:
		if('Fatal' in rules[i][0]):
			if(i in rules_dict):
				rules_dict[i] += 1
			else:
				rules_dict[i] = 1
			print(str(i) + ' -> ' +str(rules[i][0])+' '+str(rules[i][1]))

"""Print the detected association rules"""

for key, value in rules_dict.items() :
    print (key, value)