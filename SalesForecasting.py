
#%%
import sys
get_ipython().system('{sys.executable} -m pip install matplotlib pandas xlrd seaborn tqdm scikit-learn tensorflow keras protobuf')


#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, LeakyReLU, Activation
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

#%% [markdown]
# Чтение данных из Excel

#%%
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

#%% [markdown]
# Создание DataFrame для анализа зависимостей между переменными по артикулам

#%%
full_train_data = pd.DataFrame()

for x in tqdm.tqdm(sales.index[:]):
    material = x

    df = pd.concat([sales.loc[material].rename('sales'),               sales.loc[material].shift(-1).rename('sales_1'),               sales.loc[material].shift(-2).rename('sales_2'),               sales.loc[material].shift(-12).rename('sales_12'),               sales.loc[material].subtract(sales.loc[material].shift(-1)).rename('sales_diff_1'),               sales.loc[material].subtract(sales.loc[material].shift(-2)).rename('sales_diff_2'),               sales.loc[material].subtract(sales.loc[material].shift(-12)).rename('sales_diff_12'),               stock.loc[material].shift(-1).rename('stock_1'),               stock.loc[material].shift(-2).rename('stock_2'),               stock.loc[material].subtract(stock.loc[material].shift(-1)).rename('stock_diff_1'),               stock.loc[material].subtract(stock.loc[material].shift(-2)).rename('stock_diff_2'),               price.loc[material].shift(-1).rename('price_1'),               price.loc[material].shift(-1).rename('price_2'),               price.loc[material].subtract(price.loc[material].shift(-1)).rename('price_diff_1'),               price.loc[material].subtract(price.loc[material].shift(-1)).rename('price_diff_2'),               ap.loc[material].shift(-1).rename('ap_1'),               ap.loc[material].shift(-1).rename('ap_2'),               ap.loc[material].subtract(ap.loc[material].shift(-1)).rename('ap_diff_1'),               ap.loc[material].subtract(ap.loc[material].shift(-1)).rename('ap_diff_2')],              axis=1)

    df = df.dropna()

    df.loc[:,'material'] = material
    
    full_train_data = full_train_data.append(df)
    
full_train_data.loc[:,'month'] = full_train_data.index.month


#%%
del sales
del ap
del stock
del price
del hierarchy
del sell

#%% [markdown]
# Сохраняем собранные данные в файл для ускорения в будущем

#%%
full_train_data.to_excel('full_train_data.xlsx')

#%% [markdown]
# Выгружаем данные для обучения из файла

#%%
full_train_data = pd.read_excel('full_train_data.xlsx')

#%% [markdown]
# Создаем DF корелляции

#%%
df_columns_filter = ['sales','sales_1','sales_2','sales_12',
                     'sales_diff_1','sales_diff_2','sales_diff_12',
                     'stock_1','stock_2',
                     'stock_diff_1','stock_diff_2',
                     'price_1','price_2',
                     'ap_1','ap_2',
                     'month','material']


#%%
corr_df = full_train_data[df_columns_filter].groupby('material').corr().abs().sales.fillna(0).unstack(level=0).T

#%% [markdown]
# Выделение групп артикулов для ращного прогнозирования на основе коррелляции

#%%
#Cluster the data
kmeans = KMeans(n_clusters=5, random_state=0).fit(corr_df)
labels = kmeans.labels_

#Glue back to originaal data
corr_df['clusters'] = labels

# Создание нормализованной DF
corr_overal_df = corr_df.groupby('clusters').mean()

corr_overal_norm_df = (corr_overal_df-corr_overal_df.min())/(corr_overal_df.max()-corr_overal_df.min())

#%% [markdown]
# Вычисление колонок у кластера для использования для прогнозирования

#%%
# Будем отсекать по средней корелляции, за исключением артикулов, у которых нет данных
threshold = corr_overal_df[corr_overal_df.sales > 0.1].mean()
# Инициализация копии DF
corr_thresh_df = corr_overal_df.copy()
# Прогон сравнения по всем колонкам
for col in corr_overal_df.columns:
    corr_thresh_df[col] = (corr_overal_df[col] >= threshold[col]).astype(int)
    
corr_thresh_df.style.background_gradient(cmap='viridis')


#%%
def get_cluster_id(material_id):
    return corr_df.loc[material_id, 'clusters']

#%% [markdown]
# Создаем DF для всех материалов для обучения нейронки

#%%
train_data = full_train_data[df_columns_filter].merge(corr_df.clusters.to_frame(), how='left', left_on='material', right_index=True)

#%% [markdown]
# Создаем тренировочные данные

#%%
X = train_data[[col for col in train_data.columns if col not in ['sales', 'material']]]

Y = train_data.sales.to_frame()

scaler = MinMaxScaler()

print(scaler.fit(X))
print(scaler.fit(Y))

xscale=scaler.transform(X)
yscale=scaler.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


#%%
# 82.6%
model = Sequential()
model.add(Dense(32, input_dim=16, kernel_initializer='normal'))
model.add(Dense(16))
model.add(Dense(1))
model.summary()


#%%
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


#%%
history = model.fit(X_train, y_train, epochs=200, batch_size=50, validation_data=(X_test, y_test), shuffle=True, verbose=1)


#%%
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


#%%
model.evaluate(X_test, y_test)


#%%
prediction = model.predict(X_test)

accuracy = 1 - sum(abs(scaler.inverse_transform(prediction) - scaler.inverse_transform(y_test))) / sum(scaler.inverse_transform(y_test))

print('Total accuracy = %.2f%%' % (accuracy * 100))


