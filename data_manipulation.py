import pickle
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

#Getting data in form of a dictionary from a pickle file
with open('dataset.pickle', 'rb') as f:
    data = pickle.load(f)

#Loop thru each variable in dictionay and append it to its own row
rows = []

for userID, userData in data.items():
    for inputMode, userData in userData.items():
        for deviceType, userData in userData.items():
            for contentID, userData in userData.items():

                #Extract all needed data
                inputType = userData.get('input_type', str)
                inputContent = userData.get('input_content', str)
                sensorData = userData.get('data', {})
                #print(sensorData)

                ts = sensorData.get('ts', [])
                rawposX = sensorData.get('rawposX', [])
                rawposY = sensorData.get('rawposY', [])
                relposX = sensorData.get('relposX', [])
                relposY = sensorData.get('relposY', [])
                velX = sensorData.get('velX', [])
                velY = sensorData.get('velY', [])
                magX = sensorData.get('magX', [])
                magY = sensorData.get('magY', [])
                magZ = sensorData.get('magZ', [])
                orientation = sensorData.get('orientation', [])
                pressure = sensorData.get('pressure', [])
                size = sensorData.get('size', [])

                #Append data to rows
                for i in range(len(ts)):
                    rows.append({
                        'userID': userID,
                        'inputMode': inputMode,
                        'contentID': contentID,
                        'inputType': inputType,
                        'inputContent': inputContent,
                        'ts': ts[i],
                        'rawposX': rawposX[i] if i < len(rawposX) else None,
                        'rawposY': rawposY[i] if i < len(rawposY) else None,
                        'relposX': relposX[i] if i < len(relposX) else None,
                        'relposY': relposY[i] if i < len(relposY) else None,
                        'velX': velX[i] if i < len(velX) else None,
                        'velY': velY[i] if i < len(velY) else None,
                        'magX': magX[i] if i < len(magX) else None,
                        'magY': magY[i] if i < len(magY) else None,
                        'magZ': magZ[i] if i < len(magZ) else None,
                        'orientation': orientation[i] if i < len(orientation) else None,
                        'pressure': pressure[i] if i < len(pressure) else None,
                        'size': size[i] if i < len(size) else None
                    })

#Convert rows to pandas dataframe
df = pd.DataFrame(rows)

#Check if there are missing values in the data set
df_nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False), columns=['Number of Missing Values'])
df_nulls['% Missing'] = df.isnull().sum().sort_values(ascending=False)/len(df)
print(df_nulls)

msno.bar(df)
plt.show() 
