# %%
from pygam import LinearGAM, s, f, l
import pandas as pd
import patsy as pt
import numpy as np
from plotly import subplots
import plotly.offline as py
import plotly.graph_objs as go

# Prep the dataset
#   Put this back on one line!!
data = pd.read_csv( "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv" )

# %%
# Generate x and y matrices
eqn = """trips ~ -1 + month + day + hour"""
y,x = pt.dmatrices( eqn, data=data )

# Initialize and fit the model
model = LinearGAM( s(0) + s(1) + s(2) )
modelFit = model.gridsearch( np.asarray(x), y )

modelFit._is_fitted = True

# %%
# Specify plot titles and shape
titles = [ 'month', 'day', 'hour' ]

fig = subplots.make_subplots(rows = 1, cols = 3, 
	subplot_titles = titles )
fig['layout'].update( height = 800, width = 1200, 
	title = 'pyGAM', showlegend = False )

# %%
for i, title in enumerate( titles ):
  XX = modelFit.generate_X_grid( term = i )
  pdep, confi = modelFit.partial_dependence( term = i, width = .95 )
  trace = go.Scatter( x = XX[:,i], y = pdep, mode = 'lines', name = 'Effect')
  ci1 = go.Scatter( x = XX[:,i], y = confi[:,0], 
  	line = dict(dash ='dash', color = 'grey'), 
    	name='95% CI' )
  ci2 = go.Scatter( x = XX[:,i], y = confi[:,1], 
  	line = dict(dash = 'dash', color = 'grey'), 
    name = '95% CI')

  if i<3:
    fig.append_trace( trace, 1, i+1 )
    fig.append_trace( ci1, 1, i+1 )
    fig.append_trace( ci2, 1, i+1 )
  else:
    fig.append_trace( trace, 2, i-2 )
    fig.append_trace( ci1, 2, i-2 )
    fig.append_trace( ci2, 2, i-2 )
    
py.plot( fig )

# %%
dataNEW = pd.read_csv( "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv" )

#use the model to predict the number of trips using columns month, day, and hour
pred = modelFit.predict( dataNEW[['month', 'day', 'hour']] )



