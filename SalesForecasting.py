#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install matplotlib pandas xlrd seaborn tqdm scikit-learn tensorflow keras protobuf')


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


# Чтение данных из Excel

# In[3]:


sales = pd.read_excel('ДанныеДекабря_2017.xlsx', 'Продажи')
sales = sales.set_index('Material')

ap = pd.read_excel('ДанныеДекабря_2017.xlsx', 'Магазины')
ap = ap.set_index('Material')

stock = pd.read_excel('ДанныеДекабря_2017.xlsx', 'Остатки')
stock = stock.set_index('Material')

price = pd.read_excel('ДанныеДекабря_2017.xlsx', 'Цены')
price = price.set_index('Material')

hierarchy = pd.read_excel('hierarchy.xlsx')

sell = pd.read_excel('Акции.xlsx')


# In[4]:


sales.head()


# Создание DataFrame для анализа зависимостей между переменными по артикулам

# In[5]:


full_train_data = pd.DataFrame()

for x in tqdm.tqdm(sales.index[:]):
    material = x

    df = pd.concat([sales.loc[material].rename('sales'),               sales.loc[material].shift(-1).rename('sales_1'),               sales.loc[material].shift(-2).rename('sales_2'),               sales.loc[material].shift(-12).rename('sales_12'),               sales.loc[material].subtract(sales.loc[material].shift(-1)).rename('sales_diff_1'),               sales.loc[material].subtract(sales.loc[material].shift(-2)).rename('sales_diff_2'),               sales.loc[material].subtract(sales.loc[material].shift(-12)).rename('sales_diff_12'),               stock.loc[material].shift(-1).rename('stock_1'),               stock.loc[material].shift(-2).rename('stock_2'),               stock.loc[material].subtract(stock.loc[material].shift(-1)).rename('stock_diff_1'),               stock.loc[material].subtract(stock.loc[material].shift(-2)).rename('stock_diff_2'),               price.loc[material].shift(-1).rename('price_1'),               price.loc[material].shift(-1).rename('price_2'),               price.loc[material].subtract(price.loc[material].shift(-1)).rename('price_diff_1'),               price.loc[material].subtract(price.loc[material].shift(-1)).rename('price_diff_2'),               ap.loc[material].shift(-1).rename('ap_1'),               ap.loc[material].shift(-1).rename('ap_2'),               ap.loc[material].subtract(ap.loc[material].shift(-1)).rename('ap_diff_1'),               ap.loc[material].subtract(ap.loc[material].shift(-1)).rename('ap_diff_2')],              axis=1)

    df = df.dropna()

    df.loc[:,'material'] = material
    
    full_train_data = full_train_data.append(df)
    
full_train_data.loc[:,'month'] = full_train_data.index.month


# Создаем DF корелляции

# In[16]:


df_columns_filter = ['sales','sales_1','sales_2','sales_12',
                     'sales_diff_1','sales_diff_2','sales_diff_12',
                     'stock_1','stock_2',
                     'stock_diff_1','stock_diff_2',
                     'price_1','price_2',
                     'ap_1','ap_2',
                     'month','material']


# In[17]:


corr_df = full_train_data[df_columns_filter].groupby('material').corr().abs().sales.fillna(0).unstack(level=0).T


# Выделение групп артикулов для ращного прогнозирования на основе коррелляции

# In[18]:


#Cluster the data
kmeans = KMeans(n_clusters=5, random_state=0).fit(corr_df)
labels = kmeans.labels_

#Glue back to originaal data
corr_df['clusters'] = labels

# Создание нормализованной DF
corr_overal_df = corr_df.groupby('clusters').mean()

corr_overal_norm_df = (corr_overal_df-corr_overal_df.min())/(corr_overal_df.max()-corr_overal_df.min())


# Отображение кластеров и кластеров

# In[19]:


corr_overal_df.style.background_gradient(cmap='viridis')


# In[20]:


corr_overal_norm_df.style.background_gradient(cmap='viridis')


# Вычисление колонок у кластера для использования для прогнозирования

# In[21]:


# Будем отсекать по средней корелляции, за исключением артикулов, у которых нет данных
threshold = corr_overal_df[corr_overal_df.sales > 0.1].mean()
# Инициализация копии DF
corr_thresh_df = corr_overal_df.copy()
# Прогон сравнения по всем колонкам
for col in corr_overal_df.columns:
    corr_thresh_df[col] = (corr_overal_df[col] >= threshold[col]).astype(int)
    
corr_thresh_df.style.background_gradient(cmap='viridis')


# In[22]:


def get_cluster_id(material_id):
    return corr_df.loc[material_id, 'clusters']


# Создаем DF для всех материалов для обучения нейронки

# In[24]:


train_data = full_train_data[df_columns_filter].merge(corr_df.clusters.to_frame(), how='left', left_on='material', right_index=True)


# Создаем тренировочные данные

# In[62]:


X = train_data[[col for col in train_data.columns if col not in ['sales', 'material']]]

Y = train_data.sales.to_frame()

scaler = MinMaxScaler()

print(scaler.fit(X))
print(scaler.fit(Y))

xscale=scaler.transform(X)
yscale=scaler.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


# In[144]:


model = Sequential()
model.add(Dense(16, input_dim=16, kernel_initializer='normal'))
model.add(Dense(16))
model.add(Dense(1))
model.summary()


# In[145]:


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


# In[ ]:


history = model.fit(X_train, y_train, epochs=100, batch_size=50,  verbose=1, validation_split=0.2)


# In[147]:


print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[148]:


model.evaluate(X_test, y_test, batch_size=50)


# In[149]:


prediction = model.predict(X_test)


# In[150]:


scaler.inverse_transform(prediction[:10])


# In[151]:


scaler.inverse_transform(y_test[:10])


# In[ ]:




