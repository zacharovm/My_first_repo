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

### Часть 1. EDA ###

#1. Загружаем датасет
df = pd.read_csv(r"C:\Users\zacha\OneDrive\Desktop\dataset.csv")
df.drop(columns='Unnamed: 32', inplace=True)

#2. Базовые статистики датасета
#print(df.describe())
#print(df.info())

#3. Гистограммы/распределения признаков

# 1-й пример scater plot
#df.plot.scatter(x = 'radius_mean', y = 'texture_mean')
#plt.title("Зависимость radius_mean и texture_mean")
#plt.xlabel("radius_mean")
#plt.ylabel("texture_mean")
#plt.show()

# 2-й пример scater plot
#df.plot.scatter(x = 'area_mean', y = 'symmetry_mean', c = ['red'])
#plt.title("Зависимость radius_mean и texture_mean")
#plt.xlabel("area_mean")
#plt.ylabel("symmetry_mean")
#plt.show()

# гистограмма
#plt.hist(df['perimeter_mean'])
#plt.title('Гистограмма распределения perimeter_mean')
#plt.xlabel('Значения perimeter_mean')
#plt.ylabel('Частота')
#plt.show()

# гистограмма с учетом целевой переменной (M и В)
#df_m = df.loc[df['diagnosis'] == 'M']['perimeter_mean']
#df_b = df.loc[df['diagnosis'] == 'B']['perimeter_mean']
#plt.hist([df_m, df_b], color=['red', 'blue'], label=['Malignant', 'Benign'])
#plt.title('Гистограмма распределения данных по диагнозу')
#plt.xlabel('Значения')
#plt.ylabel('Частота')
#plt.show()

# гистограмма распределения M и В
#df['diagnosis'].value_counts().plot.bar(figsize=(6, 4))
#plt.grid(axis='y')
#plt.show()

# гистограмма всех признаков
#df.hist(figsize=(18, 10))
#plt.show()


# 4. heatmap для матрицы корреляций

#plt.figure(figsize=(50, 50))
#sns.heatmap(df.drop(['diagnosis'],
#                    axis=1).corr(),
#                    annot=True,
#                    cbar_kws={'orientation': 'vertical'},
#                    cbar=False,
#                    linewidths=.5,
#                    fmt='.2f')
#tick_labels = [range(0,24)]
#plt.title('Confusion Matrix')
#plt.show()

#Самые скоррелированные признаки:
# radius_mean & perimeter_mean
# area_mean & radius_mean
# perimeter_mean & area_mean
# также сильно скореелированы признаки se и worst у radius, area и perimeter.

# 5. Попарные scatterplot-ы для сильно скоррелированных признаков
# 1) radius_mean & perimeter_mean:
#df.plot.scatter(x = 'radius_mean', y = 'perimeter_mean', c = ['blue'])
#plt.title("Зависимость radius_mean и perimeter_mean")
#plt.xlabel("radius_mean")
#plt.ylabel("perimeter_mean")
#plt.show()

# 2) area_mean & radius_mean:
#df.plot.scatter(x = 'area_mean', y = 'radius_mean', c = ['blue'])
#plt.title("Зависимость area_mean и radius_mean")
#plt.xlabel("area_mean")
#plt.ylabel("radius_mean")
#plt.show()

# 3) perimeter_mean & area_mean
#df.plot.scatter(x = 'perimeter_mean', y = 'area_mean', c = ['blue'])
#plt.title("Зависимость perimeter_mean и area_mean")
#plt.xlabel("perimeter_mean")
#plt.ylabel("area_mean")
#plt.show()

# Вывод: показатели radius_mean, perimeter_mean и area_mean сильно линейно скоррелированы.

# 6. Boxplot-ы

#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(df['area_mean'])
#plt.show()

#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(df['radius_mean'])
#plt.show()

#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(df['perimeter_mean'])
#plt.show()

### Часть 2. Моделирование при помощи kNN ###

# 1. Разбейте данные на train-test, отложив 30% выборки для тестирования

# предварительно удалим ненужный столбец id
del df['id']

# посмотрим на пропорции целевой переменной:
#a = df['diagnosis'].value_counts(normalize=True)
#print(a)

#Перекодируем целевую: B - это 1; М - это 0
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'B' else 0)

# непосредственно делим выборку на test и train:
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    df.drop(['diagnosis'], axis=1), df['diagnosis'], test_size=0.30, random_state=42, stratify=df['diagnosis']
)

#print(y_train)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 2. Приведите все непрерывные переменные к одному масштабу при помощи стандартизации
# Зачем нужна стандартизация:
# - Повышение точности модели (т.к. один масштаб признаков);
# - Упрощение интерпретации результатов;
# - Ускорение процесса обучения (сейчас не актуально, но при очень больших датасетах будет важно)
# Есть и другие преимущества стандартизации. Погуглил - ее почти всегда на практике используют.

#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# 3. Постройте модель kNN "из коробки" без настройки параметров
#knn = KNeighborsClassifier(n_neighbors=2)
#knn.fit(X_train_scaled, y_train)
#accuracy = knn.score(X_test_scaled, y_test)
#print("Точность модели:", accuracy) #0,96
#y_pred = knn.predict(X_test_scaled)
#precision = precision_score(y_test, y_pred)
#print("Метрика Precision:", precision) #0,99
#recall = recall_score(y_test, y_pred)
#print("Метрика Recall:", recall) #0,94
#f1_score = f1_score(y_test, y_pred)
#print("Метрика F1:", f1_score) #0,97

#ROC-AUC
#y_pred_prob = knn.predict_proba(X_test_scaled)
#fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#auc_value = auc(fpr, tpr)
#print("Площадь под ROC-кривой:", auc_value)

# Таким образом, метрики качества базовой модели следующие:
#accuracy = 0,96
#precision = 0,99
#recall = 0,94
#f1_score = 0,97
#roc-auc = 0,99

# 4. Теперь проведите настройку параметра числа соседей на кросс-валидации.

from sklearn.model_selection import cross_val_score
#neighbors = range(1, 50)
#f1_score_train = []
#f1_score_test = []
#for k in neighbors:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    knn.fit(X_train_scaled, y_train)
#    f1_score_train.append(f1_score(knn.predict(X_train_scaled), y_train))
#    f1_score_test.append(f1_score(knn.predict(X_test_scaled), y_test))
#plt.plot(neighbors, f1_score_train, color='blue', label='train')
#plt.plot(neighbors, f1_score_test, color='red', label='test')
#plt.title("Max test quality: {:.3f}\nBest k: {}".format(max(f1_score_test), np.argmax(f1_score_test)+1))
#plt.legend()
#plt.show()

# Max f1 = 0,982 при k = 12
# В базовой модели (без кросс-валидации) f1 был равен 0,97. Точность повысилась.