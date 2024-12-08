import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

# Удаляем столбец torque
df_train = df_train.drop(['torque'], axis=1)
df_test = df_test.drop(['torque'], axis=1)

# Убираем единицы измерения из тренировочного датасета
df_train['mileage'] = df_train['mileage'].str.replace(r'[^\d\.]', '', regex=True).replace('', float('nan')).astype(float)
df_train['engine'] = df_train['engine'].str.replace(r'[^\d\.]', '', regex=True).replace('', float('nan')).astype(float)
df_train['max_power'] = df_train['max_power'].str.replace(r'[^\d\.]', '', regex=True).replace('', float('nan')).astype(float)

# Убираем единицы измерения из тестового датасета
df_test['mileage'] = df_test['mileage'].str.replace(r'[^\d\.]', '', regex=True).replace('', float('nan')).astype(float)
df_test['engine'] = df_test['engine'].str.replace(r'[^\d\.]', '', regex=True).replace('', float('nan')).astype(float)
df_test['max_power'] = df_test['max_power'].str.replace(r'[^\d\.]', '', regex=True).replace('', float('nan')).astype(float)

# Заполняем пропуски медианой в тренировочном датасете
df_train['mileage'] = df_train['mileage'].fillna(df_train['mileage'].median())
df_train['engine'] = df_train['engine'].fillna(df_train['engine'].median())
df_train['max_power'] = df_train['max_power'].fillna(df_train['max_power'].median())
df_train['seats'] = df_train['seats'].fillna(df_train['seats'].median())

# Заполняем пропуски медианой в тестовом датасете
df_test['mileage'] = df_test['mileage'].fillna(df_train['mileage'].median())
df_test['engine'] = df_test['engine'].fillna(df_train['engine'].median())
df_test['max_power'] = df_test['max_power'].fillna(df_train['max_power'].median())
df_test['seats'] = df_test['seats'].fillna(df_train['seats'].median())

# Удаляем дубликаты
describing = df_train.drop('selling_price', axis=1)
df_train = df_train.drop_duplicates(subset=describing, keep='first')
df_train = df_train.reset_index(drop=True)

df_train['engine'] = df_train['engine'].astype(int)
df_train['seats'] = df_train['seats'].astype(int)
df_test['engine'] = df_test['engine'].astype(int)
df_test['seats'] = df_test['seats'].astype(int)

# Создание датафрейма только с вещественными признаками
df_train_num = df_train.select_dtypes([int, float])
df_test_num = df_test.select_dtypes([int, float])

y_train = df_train_num['selling_price']
X_train = df_train_num.drop('selling_price', axis=1)
y_test = df_test_num['selling_price']
X_test = df_test_num.drop('selling_price', axis=1)

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaler = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaler = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Добавим категориальные признаки
cat_cols_train = df_train[['fuel', 'seller_type', 'transmission', 'owner']]
cat_cols_test = df_test[['fuel', 'seller_type', 'transmission', 'owner']]
X_train_all = pd.concat([X_train_scaler, cat_cols_train], axis=1)
X_test_all = pd.concat([X_test_scaler, cat_cols_test], axis=1)

# Применим OneHot-кодирование
cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_all_ohe = pd.DataFrame(
    ohe.fit_transform(X_train_all[cols]),
    columns=ohe.get_feature_names_out(cols),
    index=X_train_all.index)
X_test_all_ohe = pd.DataFrame(
    ohe.transform(X_test_all[cols]),
    columns=ohe.get_feature_names_out(cols),
    index=X_test_all.index)

# Удалим оригинальные столбцы из датасетов
X_train_all = X_train_all.drop(columns=cols)
X_test_all = X_test_all.drop(columns=cols)

# Добавим закодированные столбцы обратно в датасеты
X_train_all = pd.concat([X_train_all, X_train_all_ohe], axis=1)
X_test_all = pd.concat([X_test_all, X_test_all_ohe], axis=1)

ridge = Ridge(alpha=10)
ridge.fit(X_train_all, y_train)
score_ridge = ridge.score(X_test_all, y_test)