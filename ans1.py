from folium.plugins import HeatMap 
import folium
import chart_studio.plotly as py 
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import os 
files = sorted(os.listdir(r'../input/uber-pickups-in-new-york-city'))[-7:]

files.remove('uber-raw-data-janjune-15.csv')
all_data = pd.DataFrame()

for file in files:
    df = pd.read_csv('../input/uber-pickups-in-new-york-city/' + file)
    all_data = pd.concat([all_data, df])

convert_todatetime = all_data.copy()
convert_todatetime['Date/Time'] = pd.to_datetime(convert_todatetime['Date/Time'])
# Augment data with additional column
all_data['Day of week'] = convert_todatetime['Date/Time'].dt.day_name()
all_data['Day'] = convert_todatetime['Date/Time'].dt.day
all_data['Hour'] = convert_todatetime['Date/Time'].dt.hour
all_data['Minute'] = convert_todatetime['Date/Time'].dt.minute
all_data['Month'] = convert_todatetime['Date/Time'].dt.month


px.bar(x=all_data['Day of week'].value_counts().index,
      y=all_data['Day of week'].value_counts(),
     labels={'x': 'Day of week', 'y':'Counts'})

for i,month in enumerate(all_data['Month'].unique()):
    print(str(i) + " i")
    print(str(month) + " month")

# Getting the index where the month is equal to 9

plt.figure(figsize=(40,20))

for i,month in enumerate(all_data['Month'].unique()):
    plt.subplot(2,3,i+1)
    plt.xlabel('Certain Hours in Month {} '.format(month) ,fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    all_data[all_data['Month']==month]['Hour'].hist()
num_of_rides_in_months = all_data.groupby('Month')['Hour'].count()
trace1=go.Bar(
    x=all_data.groupby('Month')['Hour'].count().index,
    y=all_data.groupby('Month')['Hour'].count(),
    name='Priority'
)
iplot([trace1])
sns.distplot(all_data['Day'])
plt.figure(figsize=(10,8))
plt.hist(all_data['Day'], bins=31, rwidth=0.8) # bins is the number of equal width in the graphs
plt.xlabel('Date of the month')
plt.ylabel('Total Journeys')
plt.title('Journey by month day')

plt.figure(figsize=(20,10))

for i, month in enumerate(all_data['Month'].unique(),1): 
    plt.subplot(3,2,i)
    df_out = all_data[all_data['Month']==month]
    plt.hist(df_out['Day'])
    plt.xlabel('Day of month {}'.format(month))
    plt.ylabel('Total rides')


ax = sns.pointplot(x='Hour', y='Lat',data=all_data, hue='Day of week')
ax.set_title('offday vs Base of passenger')

base = all_data.groupby(['Base', 'Month'])['Date/Time'].count().reset_index()

plt.figure(figsize=(10,6))

sns.lineplot(x='Month', y='Date/Time', data=base, hue='Base')
dayofweek_hour = all_data.groupby(['Day of week', 'Hour']).size()
pivot = dayofweek_hour.unstack()
plt.figure(figsize=(15,6))
sns.heatmap(pivot)
def heatmap(col1, col2):
    by_cross = all_data.groupby([col1, col2]).size()
    pivot = by_cross.unstack()
    plt.figure(figsize=(15,6))
    return sns.heatmap(pivot)


heatmap('Hour', 'Day')
heatmap('Month', 'Day')
heatmap('Day of week', 'Month')
df_out = all_data[all_data['Day of week']=='Friday']
group = df_out.groupby(['Lat','Lon'])['Day of week'].count().reset_index()
coordinates = all_data.groupby(['Lat', 'Lon'])['Day of week'].count().reset_index()
coordinates.columns=['Lat', 'Lon', 'no of trips']
basemap = folium.Map()
HeatMap(coordinates,zoom=20,radius=15).add_to(basemap)
print(basemap)
def plot(df, day):
    basemap = folium.Map()
    df_out = all_data[all_data['Day of week']==day]
    HeatMap(df_out.groupby(['Lat', 'Lon'])['Day of week'].count().reset_index(),zoom=20,radius=15).add_to(basemap)
    return basemap
plot(all_data,'Friday')
uber_janjune = pd.read_csv('../input/uber-pickups-in-new-york-city/uber-raw-data-janjune-15.csv')
uber_janjune['Pickup_date'] = pd.to_datetime(uber_janjune['Pickup_date'])
uber_janjune['Day of week'] = uber_janjune['Pickup_date'].dt.day_name()
uber_janjune['Day'] = uber_janjune['Pickup_date'].dt.day
uber_janjune['Month'] = uber_janjune['Pickup_date'].dt.month
uber_janjune['Hour'] = uber_janjune['Pickup_date'].dt.hour
data_janjune = px.bar(x=uber_janjune['Month'].value_counts().index,
               y=uber_janjune['Month'].value_counts(),
            labels={'x': 'Month', 'y':'Counts'})
plt.figure(figsize=(12,6))
sns.countplot(x="Hour",data=uber_janjune) 
uber_rush_day_hour = uber_janjune.groupby(['Day of week', 'Hour'])['Pickup_date'].count().reset_index()
uber_rush_day_hour.columns=['Day of week','Hour','Trips']
plt.figure(figsize=(12,6))
sns.pointplot(x='Hour',y='Trips',hue='Day of week',data=uber_rush_day_hour)
uber_foil = pd.read_csv('../input/uber-pickups-in-new-york-city/Uber-Jan-Feb-FOIL.csv')
most_vehicles = uber_foil.groupby('dispatching_base_number')['active_vehicles'].sum()
sns.boxplot(x='dispatching_base_number',y='active_vehicles',data=uber_foil)
plt.figure(figsize=(20,10))
most_trips = uber_foil.groupby('dispatching_base_number')['trips'].sum()
sns.boxplot(x='dispatching_base_number',y='trips',data=uber_foil)
plt.figure(figsize=(20,10))
plt.figure(figsize=(12,6))
uber_foil['trips/vehicle'] = uber_foil['trips']/uber_foil['active_vehicles']
uber_foil.set_index('date').groupby(['dispatching_base_number'])['trips/vehicle'].plot()
plt.xlabel('date')
plt.title('Demand vs Supply Chart')
plt.ylabel('Avg trips/vehicle')
plt.legend()