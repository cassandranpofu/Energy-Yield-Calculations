#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Calculation of estimated energy yield for south Facing FPVs In Germany
import pandas as pd
from sklearn.linear_model import LinearRegression

# Loading data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Germany.csv')  

# Known data points-average irradiance for the 3 reference points , with the highest, lowest and medium values
#Calculation for south facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 971.0169982910156],
    'Energy_Yield': [907.115 , 814.560 , 811.386 ]  # Ensure all values are in the same unit
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Renaming the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Applying the model to the data
df['estimated_yield_south'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Updated_south.csv', index=False)  # path to save the estimated energy yield for south facing FPV.


# In[4]:


#Calculating the estmated yield for east facing FPVs in Germany

# Loading data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Updated_south.csv')  

# Known data points-average irradiance for the 3 reference points , with the highest, lowest and medium values
#Calculation for east facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 971.0169982910156],
    'Energy_Yield': [907.115 , 814.560 , 735.386 ]  # Ensure all values are in the same unit
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield_east'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Updated_east.csv', index=False)  # path for saving the results


# In[5]:


#calculating for west facing FPVs in Germany

# Loading data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Updated_east.csv')  

# Known data points-average irradiance for the 3 reference points , with the highest, lowest and medium values
#Calculating for west facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 971.0169982910156],
    'Energy_Yield': [908.209  , 809.374  , 727.459  ]  # values are in kWh/year
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield_west'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Updated_west_final.csv', index=False)  # Path for saving the results


# In[6]:


#Calculating for legal Potentential _east facing FPV

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Updated_legal.csv')  

# Known data points
#Calculation of estimated specific energy yield for east facing FPV in the legal potential
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 971.0169982910156],
    'Energy_Yield': [907.115 , 814.560 , 735.386 ]  
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Creating a linear model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Renaming the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to the dataset
df['estimated_yield_east'] = model.predict(df[['Irradiance']])


df.to_csv('/Users/cassandrampofu/Desktop/Python/Updated_legal_east.csv', index=False)  


# In[7]:


#Calculating for west facing FPV in the legal potential

# Loading data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Updated_legal_east.csv')  

# Known data points for irradiance and their corresponding estimated energy yield.
#Calculation for west facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 971.0169982910156],
    'Energy_Yield': [908.209  , 809.374  , 727.459  ]  # Ensure all values are in the same unit
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Applying the model to the data
df['estimated_yield_west'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Updated_west_legal_final.csv', index=False)  


# In[8]:


# Calculating estimated specific yield for italy
#South Facing FPV

# Loading the data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Italy.csv')  

# Known data points for calculated estimated specific yield for other lakes.
#Calculation for south facing FPV
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],
    'Energy_Yield': [1453 , 1170 , 819.445 ]  # all in KWh
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Applying the model to your data
df['estimated_yield'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Italy_Updated.csv', index=False)  


# In[9]:


# Calculating for italy
#East Facing FPV

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Italy_Updated.csv')  

#Calculation for east facing FPV
#Assumed 90 degrees azimuth
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],
    'Energy_Yield': [1307 , 1048 , 659.661 ]  # all in KWh
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Renaming the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Applying the model to the data
df['estimated_yield_east'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Italy_east.csv', index=False)  


# In[10]:


# Calculating for italy
#West Facing FPV

# Load data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Italy_east.csv')  

# Known data points
#Calculation for east facing FPV
#Assumed 270 degrees azimuth
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],
    'Energy_Yield': [1297 , 1041 , 668.008 ]  # all in KWh
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Renaming the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Applying the model to the data
df['estimated_yield_west'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Italy_Updated_final.csv', index=False)  # 


# In[11]:


#calculate for economica potential 
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Economical_potential.csv')  

# Known data points
#Calculation for south facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 974.8519897460938],
    'Energy_Yield': [1017  , 904.390  , 811.386   ]  # Ensure all values are in the same unit
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Eco_updated.csv', index=False)  # Choose your save path


# In[12]:


#calculate for economical potential 
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Eco_updated.csv')  

# Known data points
#Calculation for east facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 974.8519897460938],
    'Energy_Yield': [907.115 , 814.560 , 735.386   ]  # Ensure all values are in the same unit
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield_east'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Eco_east.csv', index=False)  # Choose your save path


# In[13]:


#calculate for economical potential 

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Eco_east.csv')  

# Known data points
#Calculation for east facing FPV
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 974.8519897460938],
    'Energy_Yield': [908.209  , 809.374  , 727.459  ]  # Ensure all values are in the same unit
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield_west'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Python/Eco_west_fin.csv', index=False)  # Choose your save path


# In[14]:


#Linear regression model for Germany, used to predict the specific energy yield for other data points.


# Known data points for Irradiance and energy yield
data = {
    'Irradiance': [1235.6409912109375, 1103.0550537109373, 971.0169982910156],  # in kWh/m²
    'Energy_Yield': [907.115 , 814.560 , 811.386]  # in kWh
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(training_df['Irradiance'], training_df['Energy_Yield'], color='blue', label='Data Points')
plt.plot(training_df['Irradiance'], model.predict(training_df[['Irradiance']]), color='red', label='Regression Line')
plt.xlabel('Irradiance (kWh/m²)')  # Corrected units for irradiance
plt.ylabel('Energy Yield (kWh)')
plt.title('Linear Regression Model for Energy Yield Estimation')
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


#Linear Regression for italy used to predict other data points
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Known data points for Italy
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],  # in kWh/m²
    'Energy_Yield': [1453 , 1170 , 819.445]  # in kWh
}

# Create a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(training_df['Irradiance'], training_df['Energy_Yield'], color='green', label='Data Points')  # Changed to green
plt.plot(training_df['Irradiance'], model.predict(training_df[['Irradiance']]), color='purple', label='Regression Line')  # Changed to purple
plt.xlabel('Irradiance (kWh/m²)')
plt.ylabel('Energy Yield (kWh)')
plt.title('Linear Regression Analysis of Energy Yield Based on Solar Irradiance in Italy')
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


#Economical Potential Italy
# Calculating for italy
#South Facing FPV
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Italy_d.csv')  

# Known data points
#Calculation for south facing FPV
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],
    'Energy_Yield': [1453 , 1170 , 819.445 ]  # all in KWh
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Italy_eco.csv', index=False)  # Choose your save path


# In[17]:


#Economical Potential Italy
# Calculating for italy
#South Facing FPV
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Python/Italy_eco.csv')  

# Known data points
#Calculation for south facing FPV
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],
    'Energy_Yield': [1307 , 1048 , 659.661 ]  # all in KWh
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield_east'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Italy_e.csv', index=False)  # Choose your save path


# In[18]:


#Economical Potential Italy
# Calculating for italy
#South Facing FPV
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data
df = pd.read_csv('/Users/cassandrampofu/Desktop/Italy_e.csv')  

# Known data points
#Calculation for south facing FPV
data = {
    'Irradiance': [1812.0050048828123, 1444.197998046875, 926.4926504235109],
    'Energy_Yield': [1297 , 1041 , 668.008 ]  # all in KWh
}

# Creating a DataFrame for training data
training_df = pd.DataFrame(data)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(training_df[['Irradiance']], training_df['Energy_Yield'])

# Rename the prediction dataset column to match the training dataset
df.rename(columns={'_meanmean': 'Irradiance'}, inplace=True)

# Apply the model to your data
df['estimated_yield_west'] = model.predict(df[['Irradiance']])

# Save the results
df.to_csv('/Users/cassandrampofu/Desktop/Italy_w_fin.csv', index=False)  # Choose your save path

