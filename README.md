# НИС: Машинное обучение. Домашняя работа №1.

В ходе данной работы я обучала модель регрессии для предсказания стоимости автомобилей и реализововала веб-сервис для применения построенной модели на новых данных. 

Файлы с результатами работы:
- [Jupyter Notebook](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/%D0%95%D0%B4%D0%B8%D0%B3%D0%B0%D1%80%D1%8F%D0%BD__AI_HW1_Regression_with_inference_base_ipynb_.ipynb) со всеми проведенными экпериментами; 
- [Дашборд](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/df_train_report.html), сохраненный в HTML-формате;
- [Файл](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/main/model_weights.pkl) с весами модели;
- [Файл](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/app.py) с реализацией сервиса приведен в папке app, в которой также находятся сопустсвующие файлы для работы сервиса:
  -  [feature_name.txt](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/feature_name.txt)  - список конечных названий признаков;
  -  [model.joblib](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/model.joblib)  - обученная модель Ridge;
  -  [ohe.joblib](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/ohe.joblib) - OneHotEncoder для кодирования данных;
  -  [scaler.joblib](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/scaler.joblib) - StandardScaler для стандартизации данных;
  -  [predictions.csv](https://github.com/karinaaa-ed/ML_HW1_Base_Edigaryan/blob/app/predictions.csv) - файл с предсказанными значениями стоимости автомобиля после запуска predict_file.
