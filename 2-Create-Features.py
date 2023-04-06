# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:30:08 2023

@author: Jfitz
"""

import pandas as pd
#import numpy as np
#from scipy import stats
#import math
#import os
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
# =============================================================================
#3 files merged 
file = 'AllReadings.csv'
data = pd.read_csv(file)
# convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = pd.to_datetime(data['timestamp'])    
#examine data
print(data.dtypes)
print(f"no of records:  {data.shape[0]}")
print(f"no of variables: {data.shape[1]}")
print((data['id'].nunique()))
# want to drop cases that dont have 24 hours of returns                               
#aggregate date and hour and include data for 24 hour period only
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute + data['hour'] * 60
aggr = data.groupby(['date','hour']).agg({'activity': 'sum'}).reset_index()
aggr = aggr[(aggr['hour'] >= 0) & (aggr['hour'] <= 23)]
counted = aggr.groupby('date').agg({'hour' : 'count'}).reset_index()
counted = counted[counted['hour'] == 24]

final = pd.merge(data, counted[['date']], on='date', how='inner')

counts = final.groupby(['id', 'date']).count()
valid_groups = counts[counts['activity'] == 1440].reset_index()[['id', 'date']]
final = final.merge(valid_groups, on=['id', 'date'])


print(final.head())



#examine the data
def newId(idVal):
    if idVal[:5] == 'condi':
        return 'Depressive'
    elif idVal[:5] == 'patie':
        return 'Schizophrenic'
    elif idVal[:5] == 'contr':
        return 'Control'
    else:
        return '*UNKNOWN*'
  
final['Category'] = final['id'].apply(newId)

if '*UNKNOWN*' in final['Category'].values:
    print("unknowns found") 
else:
    print("All 24 hours have a category")   

    
final['counter'] = final.groupby('Category').cumcount() + 1 

 
# create a patient dictionary to map each unique ID 
patient = {id:index + 1 for index, id in enumerate(data['id'].unique())}
# map the ID col to the pateientID using the dictionary values
final['patientID'] = final['id'].map(patient)

num_records = len(final)
print(f"Number of records in dataframe: {num_records}")
 
if num_records % 1440 == 0:
   print("Number of records is divisible by 1440 with no remainder")
else:
   print("Number of records is NOT divisible by 1440")







 

 
#final.to_csv('C:/mtu/project/24HourReturns.csv', index=False) 

#examine the data
def seg(hr):
    if hr < 4:
        return '00:00-04:00'
    elif hr > 3 and hr < 8:
        return '04-08:00'
    elif hr > 7 and hr < 12:
        return '08-12:00'
    elif hr > 11 and hr < 16:
        return '12-16:00'
    elif hr > 15 and hr < 20:
        return '16-20:00'
    elif hr > 19 and hr < 24:
        return '20-24:00'    
    else:
        return '*UNKNOWN*'
  
final['segment'] = final['hour'].apply(seg)

if '*UNKNOWN*' in final['segment'].values:
    print("unknowns segments found") 
else:
    print("All ids have segments")   


grouped = final.groupby(['id','date','segment'])
segmented = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
segmented = segmented.reset_index()

segmented.columns = ['id','date','segment','f.mean', 'f.sd', 'f.propZeros']
 
segmented['class'] = segmented['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
segmented = segmented[['id','date','segment','f.mean','f.sd','f.propZeros','class']]
segmented= segmented.loc[~((segmented['f.mean'] == 0
                          ) & (segmented['f.sd'] == 0))]
#segmented = segmented.loc[~((segmented['f.propZeros'] == 0))] 

print((segmented['id'].nunique()))
grouped = final.groupby(['id','date'])
newData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
newData = newData.reset_index()
 
newData.columns = ['id','date','f.mean', 'f.sd', 'f.propZeros']
 
newData['class'] = newData['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
newData = newData[['id','date','f.mean','f.sd','f.propZeros','class']]
newData = newData.loc[~((newData['f.mean'] == 0
                          ) & (newData['f.sd'] == 0))]
#newData = newData.loc[~((newData['f.propZeros'] == 0))] 
print((newData['id'].nunique()))
# =============================================================================
def newId(idVal):
     if idVal[:5] == 'condi':
         return 'Depressive'
     elif idVal[:5] == 'patie':
         return 'Schizophrenic'
     elif idVal[:5] == 'contr':
         return 'Control'
     else:
         return '*UNKNOWN*'
   
newData['Category'] = newData['id'].apply(newId)
segmented['Category'] = segmented['id'].apply(newId) 
if '*UNKNOWN*' in newData['Category'].values:
    print("unknowns found") 
else:
    print("All 24 hours have a category")   
  
newData['counter'] = newData.groupby('Category').cumcount() + 1
segmented['counter'] = segmented.groupby('Category').cumcount() + 1
# 
# # create a patient dictionary to map each unique ID 
patient = {id:index + 1 for index, id in enumerate(newData['id'].unique())}
# # map the ID col to the pateientID using the dictionary values
newData['patientID'] = newData['id'].map(patient)
segmented['patientID'] = segmented['id'].map(patient)
# =============================================================================

print(newData)    
print(segmented) 

print("***  All 3 groups Baseline input file created for 24 hr of data only ***")
print("***  Features created for 4 houtly segments ****")
# =============================================================================
# PlOTS
#-----------------------------------------------------------------------------
newData.to_csv('C:/mtu/project/24HrFeatures.csv', index=False)
segmented.to_csv('C:/mtu/project/4HrFeatures.csv', index=False)


grouped = final.groupby(['Category', 'hour'])['activity'].mean().reset_index()
pivoted = grouped.pivot(index='hour', columns='Category', values='activity')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

plt.xticks(range(24), [f'{h:02d}' for h in range(24)])


plt.xlabel(' 24 hour period hourly')
plt.ylabel('Average of daily Activity')
plt.title('Average Activity midnight to midnight per hour by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('AverageActivityPerHour.png', dpi=300)
plt.show()




grouped = final.groupby(['Category', 'minute'])['activity'].mean().reset_index()
pivoted = grouped.pivot(index='minute', columns='Category', values='activity')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

#plt.xticks(range(1440), [f'{h:0}' for h in range(1440)])


plt.xlabel(' 24 hour period in minutes')
plt.ylabel('Average Activity')
plt.title('Average Activity per minute by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('AverageActivityPerMinute.png', dpi=300)
plt.show()

#sns.barplot(x='Category', y='activity', data=final)
#final = final.set_index('segment')
#final['seg'] = final.reindex(['00-03','04-07','08-11','12-15','16-19','20-23'])

grouped = segmented.groupby(['Category', 'segment'])['f.mean'].mean().reset_index()
pivoted = grouped.pivot(index='segment', columns='Category', values='f.mean')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')
plt.rcParams["figure.autolayout"] = True
#for i, j in zip(grouped,pivoted):
 #   plt.text(i, j+0.5, '({},{})'.format(i,j))
#plt.xticks(range(24), [f'{h:4}' for h in range(24)])
#plt.xticks(final['seg'])

plt.xlabel(' 24 hour period 4 hour segments')
plt.ylabel('Mean Activity of averages')
plt.title('Mean Activity in 4 hourly segments over 24 hours')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('MeanActivityPer4hrSegement.png', dpi=300)
plt.show()


grouped = segmented.groupby(['Category', 'segment'])['f.sd'].mean().reset_index()
pivoted = grouped.pivot(index='segment', columns='Category', values='f.sd')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')
plt.rcParams["figure.autolayout"] = True
#for i, j in zip(grouped,pivoted):
 #   plt.text(i, j+0.5, '({},{})'.format(i,j))
#plt.xticks(range(24), [f'{h:4}' for h in range(24)])
#plt.xticks(final['seg'])

plt.xlabel(' 24 hour period 4 hour segments')
plt.ylabel('Standard Deviation Activity')
plt.title('SD  Activity in 4 hourly segments over 24 hours')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('SD-ActivityPer4hrSegement.png', dpi=300)
plt.show()

grouped = segmented.groupby(['Category', 'segment'])['f.propZeros'].mean().reset_index()
pivoted = grouped.pivot(index='segment', columns='Category', values='f.propZeros')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')
plt.rcParams["figure.autolayout"] = True
#for i, j in zip(grouped,pivoted):
 #   plt.text(i, j+0.5, '({},{})'.format(i,j))
#plt.xticks(range(24), [f'{h:4}' for h in range(24)])
#plt.xticks(final['seg'])

plt.xlabel(' 24 hour period 4 hour segments')
plt.ylabel('Proportion of Zero values Activity')
plt.title('Prop of Zero vals  Activity in 4 hourly segments over 24 hours')
plt.legend()
plt.gcf().set_size_inches(12, 6)

plt.savefig('PropZeros-ActivityPer4hrSegement.png', dpi=300)
plt.show()