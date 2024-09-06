# %%
#Import statements
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px

# Read data, then set the index to be the date
# NOTE: make the file a single line!!
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

# %%
data

# %%
data['datetime'] = pd.to_datetime(data['Timestamp'], 
	format='%Y-%m-%d %H:%M:%S')
data.set_index(pd.DatetimeIndex(data['Timestamp']), 
	inplace=True)

# %%
# Plot the data
px.line(data, x='Timestamp', y='trips',
       labels = {
           'datetime' : 'Date',
           'logpm' : 'Logged Pollution Level'
       })

# %%
data

# %%
data2 = data[['Timestamp','trips']]
data2.columns = ['ds','y']

# %%
# Initialize Prophet instance and fit to data

model = Prophet(changepoint_prior_scale=0.5, daily_seasonality = True )
# Higher prior values will tend toward overfitting
#     Lower values will tend toward underfitting

modelFit = model.fit(data2)

# %%
# Create timeline for 1 year in future, 
#   then generate predictions based on that timeline

future = modelFit.make_future_dataframe(periods=744, freq =' H')
pred = modelFit.predict(future)

# %%
# Create plots of forecast and truth, 
#   as well as component breakdowns of the trends

plt = modelFit.plot(pred)
plt.savefig("prophet.png")

comp = modelFit.plot_components(pred)

# %%
pred


