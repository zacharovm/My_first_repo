import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

### Часть 1. EDA ###

#1. Загружаем датасет
df = pd.read_csv(r"C:\Users\zacha\OneDrive\Desktop\AB_NYC_2019.csv")

#2. Основные шаги работы с данными

#Базовые статистики датасета
#print(df.describe())
#print(df.info())

#Удаляем ненужные признаки
df.drop(columns= {'id', 'name', 'host_id', 'host_name', 'last_review'}, inplace=True)
#print(df.info())

# Визуализируйте базовые статистики данных

# гистограмма всех признаков
#df.hist(figsize=(18, 10))
#plt.show()

# heatmap для матрицы корреляций

#plt.figure(figsize=(50, 50))
#sns.heatmap(df.drop({'price', 'neighbourhood_group', 'neighbourhood', 'room_type'},
#                    axis=1).corr(),
#                    annot=True,
#                    cbar_kws={'orientation': 'vertical'},
#                    cbar=False,
#                    linewidths=.5,
#                    fmt='.2f')
#tick_labels = [range(0,24)]
#plt.title('Confusion Matrix')
#plt.show()

# Вывод: в целом, все признаки имеют низкую попарную корреляцию.
# на данном этапе выкинул категориальные переменные - их уже дальше преобразую

#Boxplot-ы

#fig = plt.figure(figsize =(100, 70))
#plt.boxplot(df['price'])
#plt.show()

#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(df['minimum_nights'])
#plt.show()
# есть прямо сильные выбросы

#fig = plt.figure(figsize =(100, 70))
#plt.boxplot(df['number_of_reviews'])
#plt.show()


### Часть 2. Preprocessing & Feature Engineering ###

# Предобработка данных

# Для начала поработаем с категориальными переменными:

#print(df.info())
#print(df['neighbourhood'].nunique())
#enc = OrdinalEncoder()
#df[['neighbourhood']] = enc.fit_transform(df[['neighbourhood']])
df = pd.get_dummies(df, columns=['room_type'], prefix_sep='=')
df = pd.get_dummies(df, columns=['neighbourhood_group'], prefix_sep='=')
df = pd.get_dummies(df, columns=['neighbourhood'], prefix_sep='=')
#print(df['neighbourhood'].unique())
#print(df.shape) # 48895 строк
#print(df.info())

# reviews_per_month - около 10К строк null. Скорее всего, просто не было оставлено ни одного review, т.е. = 0
#df.loc[:, 'reviews_per_month'] = df.reviews_per_month.fillna(0) - вроде логично, чтобы был 0, но по факту метрики модели хуже, чем если заполнить -999
df.loc[:, 'reviews_per_month'] = df.reviews_per_month.fillna(-999)
#print(df.info())
#print(df['minimum_nights'].describe())
#print(df.head(5))
#print(df.isna().sum().sum())


# Делим данные на test и train + сделаем шкалирование StandardScaler:
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    df.drop('price', axis=1), df.price, test_size = 0.3, random_state = 42)
#print(y_train)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#scaler = RobustScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#print(X_train)
#print(X_test)


### Часть 3. Моделирование ###

# строим базовую модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
#print(model.coef_)

# считаем метрики качества R2, MAE, MSE
r2 = r2_score(y_test, y_pred)
print(r2)
mae = mean_absolute_error(y_test, y_pred)
print(mae)
mse = mean_squared_error(y_test, y_pred)
#print(mse)
# вывод: кажется, что метрики не очень. R2 только 12%; mae - 73, при том, что средняя цена 153...
# Попробуем улучшить модель как с помощью новых признаков (или модификации старых), так и с помощью новых алгоритмов
# Что попробовал:
# 1) попробовал удалить разные текущие признаки: latitude, longitude, availability_365 - не помогло, стало хуже
#print(df.info())

#2) Разные модели применил

# Модель Ridge: - не помогло, метрики примерно такие же
#from sklearn.linear_model import Ridge
#model = Ridge(alpha = 100)
#model.fit(X_train, y_train)
#y_pred_2 = model.predict(X_test)
#y_train_pred = model.predict(X_train)
#r2_2 = r2_score(y_test, y_pred_2)
#mae_ridge = mean_absolute_error(y_test, y_pred_2)
#print(r2_2)
#print(mae_ridge)

# Модель Lasso - не помогло, метрики еще хуже стали
#from sklearn.linear_model import Lasso
#model = Lasso(alpha = 100)
#model.fit(X_train, y_train)
#y_pred_3 = model.predict(X_test)
#y_train_pred = model.predict(X_train)
#mae_lasso = mean_absolute_error(y_test, y_pred_3)
#print(mae_lasso)
#r2_3 = r2_score(y_test, y_pred_3)
#print(r2_3)

# Модель ElasticNet: сомнительно, чуть улучшилось mae, чуть упала r2
#from sklearn.linear_model import ElasticNet
#model = ElasticNet(alpha = 0.5,  l1_ratio= 0.5)
#model.fit(X_train, y_train)
#y_pred_4 = model.predict(X_test)
#y_train_pred = model.predict(X_train)
#mae_elastic_net = mean_absolute_error(y_test, y_pred_4)
#print(mae_elastic_net)
#r2_4 = r2_score(y_test, y_pred_4)
#print(r2_4)

#3) Немного улучшило результат вот это:
#df.loc[:, 'reviews_per_month'] = df.reviews_per_month.fillna(-999)
#То есть лучше заполнять пропуски на -999, чем на 0 (хотя вроде как 0 логичнее)

#4) еще улучшило результат вот это:
#df = pd.get_dummies(df, columns=['room_type'], prefix_sep='=')
#df = pd.get_dummies(df, columns=['neighbourhood_group'], prefix_sep='=')
#df = pd.get_dummies(df, columns=['neighbourhood'], prefix_sep='=')
# изначально через OrdinalEncoder прогонял
# датасет на 237 колонок выглядит устрашающе, но метрики явно улучшились: с 12 до 15,5 (r2); get_dummies работает

# 5) RobustScaler - не помог, метрики никак не улучшились по сравнению с StandardScaler


### Итоговые выводы ###

#1) Предобработка данных дает лучше результат, чем если ее не делать
#2) Какие методы улучшили метрики линейной регрессии:
# - get_dummies;
# - заполнение пустых значений числом -999 (а не 0);
# 3) Что НЕ помогло улучшить метрики:
# - RobustScaler (никак на метриках не отразилось)
# - различные модификации алгоритмов линейной регрессии (Ridge, Lasso, ElasticNet)
# - Удаление разных признаков (например, latitude, longitude, availability_365)
# 4) Бизнес-вывод: я бы не стал использовать данную модель в проде.
# Средняя цена = 153, а mae = 69 (это после всех улучшений).
# То есть разброс предсказанных и реальных цен будет очень большим, почти 50%.
# Думаю, что экспертные знания дадут более точный результат.
# 5) Какие дальнейшие шаги можно сделать, чтобы значительно улучшить метрики модели:
# - добавить еще ряд релевантных признаков (например, пообщаться с экспертами, что не учли)
# - использовать другие алгоритмы, например, градиентный бустинг и нейросети
# - использовать больше данных для обучения модели
