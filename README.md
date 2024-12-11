# НИС: Машинное обучение. Домашняя работа №1.

В ходе данной работы я обучала модель регрессии для предсказания стоимости автомобилей и реализововала веб-сервис для применения построенной модели на новых данных. 

**Файлы с результатами работы:**
- [Jupyter Notebook](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/%D0%95%D0%B4%D0%B8%D0%B3%D0%B0%D1%80%D1%8F%D0%BD__AI_HW1_Regression_with_inference_base_ipynb_.ipynb) со всеми проведенными экпериментами; 
- [Дашборд](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/df_train_report.html), сохраненный в HTML-формате;
- [Файл](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/model_weights.pkl) с весами модели;
- [Файл](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/app.py) с реализацией сервиса приведен в папке app, в которой также находятся сопустсвующие файлы для работы сервиса:
  -  [feature_name.txt](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/app/feature_names.txt)  - список конечных названий признаков;
  -  [model.joblib](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/app/model.joblib)  - обученная модель Ridge;
  -  [ohe.joblib](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/app/ohe.joblib) - $OneHotEncoder$ для кодирования данных;
  -  [scaler.joblib](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/app/scaler.joblib) - $StandardScaler$ для стандартизации данных;
  -  [predictions.csv](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/app/predictions.csv) - файл с предсказанными значениями стоимости автомобиля после запуска predict_file.

---

**Алгоритм действий, который был применен для данной работы:**
1. Приведение данных к стандартному виду:
   - заполнение пропусков,
   - удаление дубликатов,
   - удаление единиц из переменных ``mileage``, ``engine`` и ``max_power`` и приведение их к необхходимому типу данных ``float``,
   - удаление столбцов ``name`` и ``torque``, так как они не несли большой смысловой нагрузки;
2. Визуализация данных с помощью построения графиков ``pairplot`` и корреляции Пирсона;
3. Cтандартизация вещественных признаков с помощью $StandardScaler$;
4. Обучение модели сначала на вещественных признаках с помощью линейной регрессии, Lasso регрессии, ElasticNet и оценка обученных моделей метриками качества $R^2$ и $MSE$;
5. Добавление категориальных признаков и кодирование их с помощью $OneHotEncoder$;
6. Обучение модели с помощью Ridge регрессии и ее оценка метриками качества $R^2$ и $MSE$;
7. Создание собственной бизнесовой метрики (``business_metric``) для оценки качества всех обученных моделей;
8. Реализация сервиса на FastAPI.

---

Визуализация данных с помощью построения графиков ``pairplot`` позволила отследить связь признаков с целевой переменной:

На графиках видно сильную связь между стоимостью автомобиля (``selling_price``) и годом выпуска (``year``), пробегом (``km_driven``), двигателем (``engine_CC``) и максимальной мощностью (``max_power_bhp``). Графики ``selling_price``=f(``engine_CC``) и ``selling_price``=f(``max_power_bhp``) имеют почти линейную зависимость. График ``selling_price``=f(``year``) похож на график фукции $y=f(-1/x)$ при $x>0$. График ``selling_price``=f(``km_driven``) похож на график фукции $y=f(1/x)$ при $x>0$.



(https://drive.google.com/drive/folders/1Q38hXAAkNSQwRQle7VJcB_NshqZagZpo?usp=drive_link)
