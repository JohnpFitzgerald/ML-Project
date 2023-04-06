# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:06:04 2023

@author: fitzgeraldj
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def mergeLoop(files):
# Create a DataFrame to store the merged data
  mergedData = pd.DataFrame()
#mergedData = pd.read_csv("Readings.csv")    
# Loop through all CSV files in the named directory and concatenate into the new DataFrame
# and apply the filename as a new variable
  for file in os.listdir(files):
    if file.endswith('.csv'):
        # Read the CSV file into a DataFrame and add a new column to identify the file
        data = pd.read_csv(os.path.join(files, file))
        data['id'] = file.split('.')[0]  # Add a new column with the file name as identifier
        
        # Append the DataFrame to the merged_data DataFrame
        mergedData = pd.concat([mergedData, data], ignore_index=True)    
  return mergedData


def mergeDepression():
#merge the DEPRESSION files along with the control group files and apply a
#new column to identify patient/Control:
# Set the path to the directory containing the CSV files
   path = 'C:/mtu/project/depMerge/'    
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/DepReadings.csv', index=False)

def mergeSchizophrenia():
#merge the SCHIZOPHRENIA files along with the CONTROL group files
# and apply a new variable to identify Condition or Control:
# Set the path to the directory containing the CSV files
   path = 'C:/mtu/project/schMerge/'
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/SchReadings.csv', index=False) 
   
def mergeControl():
#merge the SCHIZOPHRENIA files along with the CONTROL group files
# and apply a new variable to identify Condition or Control:
# Set the path to the directory containing the CSV files
   path = 'C:/mtu/project/conMerge/'
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/ConReadings.csv', index=False) 
   
   
def merge3Files():
    Contro = 'C:/mtu/project/ConReadings.csv'
    Schizo = 'C:/mtu/project/SchReadings.csv'
    Depre = 'C:/mtu/project/DepReadings.csv' 
    print("***    Merging multiple files into a single pandas dataframe  ***")
    allData= pd.concat(map(pd.read_csv, [Contro,Schizo,Depre]), ignore_index=True)
    allData.to_csv('C:/mtu/project/AllReadings.csv', index=False)

def mergeControlDepression():
    Depre = 'C:/mtu/project/DepReadings.csv' 
    Contro = 'C:/mtu/project/ConReadings.csv'
    print("***    Merging Control & Depression files into a single pandas dataframe  ***")
    CDData = pd.concat(map(pd.read_csv, [Contro,Depre]), ignore_index=True)
    CDData.to_csv('C:/mtu/project/Control-Dep-Readings.csv', index=False)    
    return CDData

def mergeControlSchizophrenia():
    Depre = 'C:/mtu/project/SchReadings.csv' 
    Contro = 'C:/mtu/project/ConReadings.csv'
    print("***    Merging Control & Schizophrenia files into a single pandas dataframe  ***")
    CDData = pd.concat(map(pd.read_csv, [Contro,Depre]), ignore_index=True)
    CDData.to_csv('C:/mtu/project/Control-Dep-Readings.csv', index=False)    
    return CDData



#================================================================================
# Create features



def createDepFeatures():
    CDData = mergeControlDepression()
    # convert the "date" column to a datetime object
    CDData['date'] = pd.to_datetime(CDData['date'])
    CDData['timestamp'] = pd.to_datetime(CDData['timestamp'])

    grouped = CDData.groupby(['id','date'])
    newData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
    newData = newData.reset_index()

    #print(newData.columns)

    #result = newData[['id',[('activity', 'mean'), ('activity', 'std'), ('activity', '<lambda_0>')]]]

   
    #result = newData[['id', 'activity']].groupby('id').agg(['mean', 'std', lambda x: np.percentile(x, 75)-np.percentile(x, 25)])
    newData.columns = ['userid','date','f.mean', 'f.sd', 'f.propZeros']
    # Save the result to a CSV file
    newData['class1'] = newData['userid'].str[:5].apply(lambda x: 1 if x == 'condi' else 0) 
    newData = newData[['userid','date','class1','f.mean','f.sd','f.propZeros']]
    newData = newData.loc[~((newData['f.mean'] == 0) & (newData['f.sd'] == 0))]
    newData = newData.loc[~((newData['f.propZeros'] == 0))]    
    print("***   Depression Baseline input file created  ***")
    newData.to_csv('C:/mtu/project/Depression-features.csv', index=False)

def createSchFeatures():
    CDData = mergeControlSchizophrenia()
    # convert the "date" column to a datetime object
    CDData['date'] = pd.to_datetime(CDData['date'])
    CDData['timestamp'] = pd.to_datetime(CDData['timestamp'])

    grouped = CDData.groupby(['id','date'])
    newData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
    newData = newData.reset_index()

    #print(newData.columns)

    #result = newData[['id',[('activity', 'mean'), ('activity', 'std'), ('activity', '<lambda_0>')]]]

   
    #result = newData[['id', 'activity']].groupby('id').agg(['mean', 'std', lambda x: np.percentile(x, 75)-np.percentile(x, 25)])
    newData.columns = ['userid','date','f.mean', 'f.sd', 'f.propZeros']
    # Save the result to a CSV file
    newData['class1'] = newData['userid'].str[:5].apply(lambda x: 1 if x == 'patie' else 0) 
    newData = newData[['userid','date','class1','f.mean','f.sd','f.propZeros']]
    newData = newData.loc[~((newData['f.mean'] == 0) & (newData['f.sd'] == 0))]
    newData = newData.loc[~((newData['f.propZeros'] == 0))]    
    print("***   Schizophrenia Baseline input file created  ***")
    newData.to_csv('C:/mtu/project/Schizophrenia-features.csv', index=False)





def visualizeDepression():
   #current_path = os.getcwd()
   file = 'Depression-features.csv'
   #data = pd.read_csv(current_path + file)
   #JFitz - Set to read file from same directory as code
   df = pd.read_csv(file)    

   userid = 'control_3'
   data = df[df['userid'] == 'control_3']

   fig, ax = plt.subplots()

   # Plot the data
   ax.plot(data['userid'], data['f.mean'], label='Mean')

    # Set x-axis tick locations and labels
   ax.xaxis.set_major_locator(ticker.AutoLocator())
   ax.xaxis.set_major_formatter(ticker.ConciseDateFormatter(ticker.AutoDateLocator()))

   # Rotate x-axis tick labels for better readability
   plt.xticks(rotation=45)

   # Add legend and axis labels
   ax.legend()
   ax.set_xlabel('Date')
   ax.set_ylabel('Mean')
   ax.set_title(f'Mean by date for userid {userid}')

   # Show the plot
   plt.show()
   
#visualizeDepression()   
def createAllFeatures():
    #CDData = merge3Files()
    CDData = pd.read_csv('AllReadings.csv')
    CDData['date'] = pd.to_datetime(CDData['date'])
    CDData['timestamp'] = pd.to_datetime(CDData['timestamp'])


    grouped = CDData.groupby(['id','date'])
    newData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
    newData = newData.reset_index()
    #print(newData.columns)

    #result = newData[['id',[('activity', 'mean'), ('activity', 'std'), ('activity', '<lambda_0>')]]]

   
    #result = newData[['id', 'activity']].groupby('id').agg(['mean', 'std', lambda x: np.percentile(x, 75)-np.percentile(x, 25)])
    newData.columns = ['userid','date','f.mean', 'f.sd', 'f.propZeros']
    # Save the result to a CSV file
    #newData['class1'] = newData['userid'].str[:5].apply(lambda x: 1 if x == 'condi' else 0) 
    newData['class1'] = newData['userid'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
    newData = newData[['userid','date','class1','f.mean','f.sd','f.propZeros']]
    newData = newData.loc[~((newData['f.mean'] == 0) & (newData['f.sd'] == 0))]
    newData = newData.loc[~((newData['f.propZeros'] == 0))]    
    print("***  All 3 groups Baseline input file created  ***")
    newData.to_csv('C:/mtu/project/All3-features.csv', index=False)

mergeSchizophrenia() 
mergeDepression()   
mergeControl()
merge3Files()
#createDepFeatures()
#createSchFeatures()
#createAllFeatures()
