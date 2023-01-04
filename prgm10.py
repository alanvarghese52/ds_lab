from tensorflow import keras

print('Tensorflow/keras : %s' % keras.__version__)
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
import pandas as pd

print('pandas : %s' % pd.__version__)
import numpy as np

print('numpy : %s' % np.__version__)
import sklearn

print('sklearn : %s' % sklearn.__version__)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly
import plotly.express as px
import plotly.graph_objects as go

print('plotly : %s' % plotly.__version__)
pd.options.display.max_columns = 50
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')
df = df[pd.isnull(df['RainTomorrow']) == False]
# df=df.fillna(df.mean())
df['RainTodayFlag'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
df['RainTomorrowFlag'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df)
X = df[['Humidity3pm']]
Y = df['RainTomorrowFlag'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = Sequential(name="Model-with-One-Input")
model.add(Input(shape=(1,), name='Input-Layer'))
model.add(Dense(2, activation='softplus', name='Hidden-Layer'))
model.add(Dense(1, activation='sigmoid', name='Output-Layer'))
