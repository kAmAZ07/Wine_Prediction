import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

matplotlib.use('Agg')
import matplotlib.pyplot as plt


APP_DIR = Path(__file__).resolve().parent

FEATURE_NAMES = {
    'fixed_acidity': 'Нелетучая кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'Кислотность (pH)',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино'
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def predict_probabilities(_model, X):
    return _model.predict_proba(X)[:, 1]


def read_batch(content, features):
    try:
        data = pd.read_csv(BytesIO(content), encoding='utf-8-sig')
    except pd.errors.EmptyDataError:
        raise ValueError('Файл пуст: нужны заголовки и строки с признаками вина.') from None
    except pd.errors.ParserError:
        raise ValueError('Не удалось прочитать CSV. Проверьте разделители и кавычки.') from None
    except UnicodeError:
        raise ValueError('Не удалось прочитать кодировку файла. Сохраните CSV в UTF-8.') from None
    missing = [name for name in features if name not in data.columns]

    if missing:
        raise ValueError('В файле не хватает столбцов: ' + ', '.join(missing))

    if data.empty:
        raise ValueError('В файле нет строк с данными.')

    X = data.loc[:, features].apply(pd.to_numeric, errors='coerce')
    invalid = X.columns[~np.isfinite(X.to_numpy(dtype=float)).all(axis=0)]

    if len(invalid):
        raise ValueError(
            'Нужны числовые значения без пропусков и бесконечностей в столбцах: '
            + ', '.join(invalid)
        )

    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 и 1.')

    return data, X


def show_single_wine(model, X_test, features, threshold):
    st.subheader('Оценка одного вина')
    st.caption('Начальные значения — медианы тестовой таблицы. Пределы полей взяты из неё же.')

    color = st.radio(
        'Тип вина',
        ['Белое', 'Красное'],
        index=int(X_test['is_red'].median()),
        horizontal=True,
        key='wine_type'
    )

    values = {'is_red': int(color == 'Красное')}
    columns = st.columns(3)

    numeric_features = [name for name in features if name != 'is_red']

    for index, name in enumerate(numeric_features):
        if name == 'density':
            step, number_format = 0.00001, '%.5f'
        elif name == 'chlorides':
            step, number_format = 0.001, '%.3f'
        elif name in ['free_sulfur_dioxide', 'total_sulfur_dioxide']:
            step, number_format = 1.0, '%.1f'
        else:
            step, number_format = 0.01, '%.3f'

        with columns[index % 3]:
            values[name] = st.number_input(
                FEATURE_NAMES[name],
                min_value=float(X_test[name].min()),
                max_value=float(X_test[name].max()),
                value=float(X_test[name].median()),
                step=step,
                format=number_format,
                key=f'feature_{name}'
            )

    wine = pd.DataFrame([values], columns=features)
    probability = float(predict_probabilities(model, wine)[0])

    st.metric('Вероятность хорошего вина', str(probability))

    if probability >= threshold:
        st.success('Включить в подборку')
    else:
        st.info('Не включать в подборку')


def show_metrics(y_test, probabilities, threshold):
    st.subheader('Модельные метрики')

    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)

    columns = st.columns(3)
    columns[0].metric('Точность отбора', str(float(precision)))
    columns[1].metric('Полнота', str(float(recall)))
    columns[2].metric('F1', str(float(f1)))

    st.write(f'Отобрано вин: {int(predictions.sum())} из {len(y_test)}.')
    st.write(
        'Повышение порога сокращает подборку и может увеличить точность отбора, '
        'но часть хороших вин будет пропущена. Снижение порога расширяет подборку: '
        'в неё могут попасть и дополнительные хорошие вина, и лишние рекомендации.'
    )

    matrix = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 4))
    display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=['Остальные вина', 'Хорошие вина']
    )
    display.plot(ax=axis, cmap='Blues', values_format='d', colorbar=False)
    axis.set_xlabel('Предсказанный класс')
    axis.set_ylabel('Настоящий класс')
    axis.set_title('Матрица ошибок')
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(
        y_test, probabilities
    )

    tn, fp, fn, tp = matrix.ravel()
    current_fpr = fp / (fp + tn)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, label='Модель')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Случайное ранжирование')
    axes[0].scatter(current_fpr, recall, color='red', s=70, zorder=5, label='Текущий порог')
    axes[0].set_title('ROC-кривая')
    axes[0].set_xlabel('Доля ошибочного отбора среди остальных вин')
    axes[0].set_ylabel('Полнота')

    axes[1].step(recall_curve, precision_curve, where='post', label='Модель')
    axes[1].axhline(float(np.mean(y_test)), linestyle='--', color='gray', label='Случайный отбор')
    axes[1].scatter(recall, precision, color='red', s=70, zorder=5, label='Текущий порог')
    axes[1].set_title('PR-кривая')
    axes[1].set_xlabel('Полнота')
    axes[1].set_ylabel('Точность отбора')

    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.legend()
        axis.grid(True)

    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.write('ROC-AUC:', float(metrics.roc_auc_score(y_test, probabilities)))
    st.write('PR-AUC:', float(metrics.average_precision_score(y_test, probabilities)))
    st.caption('Если ничего не отобрано, точность отбора принимается равной нулю.')


def show_batch(model, X_test, features, threshold):
    st.subheader('Пакетное предсказание')
    st.write('Загрузите CSV с 12 признаками вина. Столбец target не нужен.')
    st.caption('Для всех строк используется порог из боковой панели.')

    st.download_button(
        'Скачать пример CSV',
        data=X_test.head(5).to_csv(index=False).encode('utf-8-sig'),
        file_name='wine_example.csv',
        mime='text/csv',
        key='example_download'
    )

    uploaded = st.file_uploader('Файл с признаками вин', type=['csv'], key='batch_file')

    if uploaded is None:
        return

    try:
        data, X = read_batch(uploaded.getvalue(), features)
    except ValueError as error:
        st.error(str(error))
        return

    probabilities = predict_probabilities(model, X)
    result = data.copy()
    result['Вероятность хорошего вина'] = probabilities
    result['Порог'] = threshold
    result['Решение'] = np.where(
        probabilities >= threshold,
        'Включить',
        'Не включать'
    )

    st.dataframe(result, hide_index=True, width='stretch')
    st.download_button(
        'Скачать предсказания',
        data=result.to_csv(index=False).encode('utf-8-sig'),
        file_name='wine_predictions.csv',
        mime='text/csv',
        key='predictions_download'
    )


def show_coefficients(model, features):
    st.subheader('Что выучила модель')

    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0],
        index=features
    ).sort_values()

    labels = [FEATURE_NAMES[name] for name in coefficients.index]
    colors = ['#39865b' if value > 0 else '#b84c4c' for value in coefficients]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, coefficients.to_numpy(), color=colors)
    axis.axvline(0, color='gray', linewidth=1)
    axis.set_xlabel('Коэффициент после стандартизации')
    axis.set_title('Коэффициенты логистической регрессии')
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)

    st.write('Сильнее всего голосуют за хорошее вино: ' + ', '.join(
        FEATURE_NAMES[name].lower() for name in positive.index
    ) + '.')
    st.write('Сильнее всего голосуют против: ' + ', '.join(
        FEATURE_NAMES[name].lower() for name in negative.index
    ) + '.')
    st.write(
        'Положительный коэффициент повышает оценку хорошего вина при прочих равных, '
        'отрицательный — понижает. Признаки стандартизированы, поэтому коэффициенты '
        'можно сравнивать по модулю.'
    )


def main():
    st.set_page_config(page_title='Подбор вина', page_icon='🍷', layout='wide')
    st.title('Подбор вина')
    st.write('Оценка вина по лабораторным показателям и проверка качества подборки.')

    model = load_model()
    test_data = load_test_data()
    settings = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))

    features = list(model.feature_names_in_)
    X_test = test_data.loc[:, features]
    y_test = test_data['target'].to_numpy()
    probabilities = predict_probabilities(model, X_test)

    st.sidebar.header('Настройки подборки')
    threshold = st.sidebar.slider(
        'Порог отбора',
        min_value=0.0,
        max_value=1.0,
        value=float(settings['threshold']),
        step=0.01,
        key='threshold'
    )
    st.sidebar.caption('Чем выше порог, тем меньше вин попадёт в подборку.')

    single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
        'Одно вино',
        'Модельные метрики',
        'Пакетное предсказание',
        'Что выучила модель'
    ])

    with single_tab:
        show_single_wine(model, X_test, features, threshold)

    with metrics_tab:
        show_metrics(y_test, probabilities, threshold)

    with batch_tab:
        show_batch(model, X_test, features, threshold)

    with coefficients_tab:
        show_coefficients(model, features)


if __name__ == '__main__':
    main()
