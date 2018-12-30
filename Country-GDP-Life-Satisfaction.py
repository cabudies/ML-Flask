import pandas as pd
import matplotlib.pyplot as plt

gdp_data = pd.read_csv('data/gdp_rate.csv', encoding = "ISO-8859-1")
print(gdp_data.head())

gdp_data = gdp_data[['Country', '2015']]
print(gdp_data.head())

gdp_data.rename(columns={'2015': 'GDP'}, inplace=True)
print(gdp_data.head())

life_index_data = pd.read_csv('data/life_index.csv')
print(life_index_data.head())

life_index_data = life_index_data[(life_index_data['INEQUALITY']== "TOT") & (life_index_data['Indicator']=='Life satisfaction')]
print(life_index_data.head())

life_index_data = life_index_data.groupby(['Country'])['Value'].sum()
print(life_index_data.head())

life_index_data = pd.DataFrame(life_index_data)
print(life_index_data.head())

life_index_data.rename(columns={'Value': 'Life_Satisfaction'}, inplace=True)
print(life_index_data.head())

gdp_data = gdp_data.set_index(['Country'])
print(gdp_data.head())

gdp_life_data = pd.merge(left=gdp_data, right=life_index_data, left_index=True, right_index=True)
print(gdp_life_data.head())

### Visualize the data using the scatter plot

gdp_life_data.info()

values = [x.replace(',','') for x in gdp_life_data.GDP]
print(values)

gdp_life_data.GDP = pd.to_numeric(values)
print(gdp_life_data.head())

gdp_life_data.sort_values(by='GDP', ascending=True, inplace=True)
print(gdp_life_data.head())

remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))
gdp_life_data = gdp_life_data.iloc[keep_indices][:29]
print(gdp_life_data)

print(gdp_life_data.head())

plt.figure(figsize=(20, 8))
plt.scatter(x='GDP', y='Life_Satisfaction', data=gdp_life_data)
plt.xlabel('GDP of the country')
plt.ylabel('Life Satisfaction')
plt.title('GDP-Life Satisfaction')
# plt.show()


## Performing Machine Learning - Linear Regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

from sklearn.model_selection import train_test_split

training_data = gdp_life_data['GDP']
output_data = gdp_life_data['Life_Satisfaction']

import numpy as np

train = np.c_[gdp_life_data.GDP]
output = np.c_[gdp_life_data.Life_Satisfaction]

new_train_data = gdp_life_data.GDP.values.tolist()

new_output_data = np.array(gdp_life_data.Life_Satisfaction.values)

print(new_train_data)

X, x_test, Y, y_test = train_test_split(train, output, test_size=0.2, random_state=42)

model.fit(X, Y)

prediction = model.predict([[8113]])
print(prediction)

### save your machine learning model

from sklearn.externals import joblib

file = 'country-gdp.sav'
joblib.dump(model, filename=file)
















