import os, types
import io, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sn

##################################################
# read the historical electric load data
path = 'https://raw.githubusercontent.com/WiDSTexas2021/datathon-code/main/data/ercot_hourly_load.csv'
data = requests.get(path).content
load_df = pd.read_csv(io.StringIO(data.decode('utf-8')))

# add the date_time as index
load_df['date_time'] = pd.to_datetime(load_df['Hour_Ending'].apply(lambda x: x.replace('-06:00','').replace('-05:00','')))
load_df.drop(['Hour_Ending'], axis = 1, inplace = True).set_index('date_time',inplace=True)

##################################################
# read weatherHistory data
path = 'https://raw.githubusercontent.com/WiDSTexas2021/datathon-code/main/data/weather_history.csv'
data = requests.get(path).content
weather_df = pd.read_csv(io.StringIO(data.decode('utf-8')))

# add 'date_time' column and set as index
weather_df.loc[:, 'time'] = weather_df.loc[:, 'time'].apply(lambda time: str(time).zfill(4))
weather_df['date_time'] = pd.to_datetime(weather_df['date'] + ' ' + weather_df['time'])
weather_df.drop(['date','time'],axis = 1,inplace=True).set_index('date_time')

# add zone column
ercotWeatherMap={'Abilene':'West', 'Corpus Christi':'South', 'Dallas':'North Central','Houston':'Coast', 
                 'Midland':'Far West', 'San Antonio':'South Central','Tyler':'East', 'Wichita Falls':'North'}
weather_df['zone']= weather_df['city'].map(ercotWeatherMap)
weather_df.dropna(axis = 0, inplace = True).drop(['city'], axis = 1, inplace = True)

#######################################################
# Data Exploration
# visualize the raw electrical load data
regions = ["Coast", "East", "Far West", "North", "North Central", "South", "South Central", "West"]
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12,20))
for ax, region in zip(axes.flat, regions):
    ax.plot(hr_load.index, hr_load[region])
    ax.set_title(region)

# correlation coef heat map
corr = d.corr(weather_df)
sn.heatmap(corr,center=0)
# drop the highly correlated features: heatIndexF, feelslikeF,windchillF,windgustMiles,dewPointF
weather_df = weather_df[['tempF', 'windspeedMiles','visibilityMiles', 'winddirDegree', 'winddir16Point', 'weatherDesc', 'precipMM',  
                         'humidity', 'visibility', 'pressure',  'cloudcover', 'uvIndex']]

# distributions of load and temperature 
plt.figure(figsize=(15,5))
plt.plot(load_df['West'].index, load_df['West'])
plt.plot(weather_df[weather_df['zone']=='West'].index,weather_df[weather_df['zone']=='West']['tempF'])
plt.legend(['load','tempF'],loc='best')
plt.title('West')

# two days distributions of load and temperature 
plt.figure(figsize=(15,5))
plt.plot(load_df['West'].index[-48:], load_df['West'][-48:])
plt.plot(weather_df[weather_df['zone']=='West'].index[-48:],weather_df[weather_df['zone']=='West']['tempF'][-48:])
plt.legend(['load','tempF'],loc='best')
plt.title('West')

# Comment on the exploration:
# The temperature shows highly positive correlation with the electric load. 
# All the ERCOT regions show an annual periodicity and most show a slight trend up as time progresses. 
# Looking at the data on a smaller scale also shows a daily periodicity.

