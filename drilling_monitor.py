# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import time
from datetime import datetime


# 1. Генерация и сохранение синтетических данных + обучение модели
def generate_and_train():
    # Генерация синтетических данных
    def generate_drilling_data(num_samples=1000):
        np.random.seed(42)
        data = {
            'depth': np.random.uniform(1000, 5000, num_samples),
            'pressure': np.abs(np.random.normal(35, 10, num_samples)),
            'rop': np.abs(np.random.lognormal(2, 0.5, num_samples)),
            'vibration': np.abs(np.random.weibull(1.5, num_samples)),
            'lithology': np.random.choice(['sandstone', 'shale', 'limestone'], num_samples),
            'complication': np.random.binomial(1, 0.2, num_samples)
        }
        df = pd.DataFrame(data)
        df = pd.get_dummies(df, columns=['lithology'])
        return df

    # Создание и сохранение данных
    df = generate_drilling_data()
    df.to_csv('drilling_data.csv', index=False)

    # Обучение модели
    X = df.drop('complication', axis=1)
    y = df['complication']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(model, 'drilling_model.pkl')
    return model


# 2. Streamlit интерфейс
def main():
    st.set_page_config(
        page_title="Drilling Operations Monitor",
        layout="wide",
        page_icon="⛏️"
    )

    # Загрузка или обучение модели
    try:
        model = joblib.load('drilling_model.pkl')
    except:
        st.warning("Модель не найдена, запуск обучения...")
        model = generate_and_train()
        st.success("Модель успешно обучена!")

    # Генератор реальных данных
    class DataGenerator:
        def __init__(self):
            self.last_depth = 1500.0
            self.time_points = 60

        def generate_live_data(self):
            now = datetime.now()
            timestamps = pd.date_range(
                end=now,
                periods=self.time_points,
                freq='S'
            )

            data = {
                'timestamp': timestamps,
                'depth': np.linspace(self.last_depth, self.last_depth + 30, self.time_points),
                'pressure': np.abs(np.random.normal(40, 5, self.time_points) + np.sin(
                    np.linspace(0, 4 * np.pi, self.time_points)) * 3),
                'rop': np.abs(np.random.lognormal(2.1, 0.3, self.time_points)),
                'vibration': np.abs(np.random.weibull(1.3, self.time_points)),
                'lithology': np.random.choice(['sandstone', 'shale', 'limestone'], self.time_points)
            }

            self.last_depth += 30
            return pd.DataFrame(data)

    # Инициализация генератора
    data_gen = DataGenerator()

    # Основной интерфейс
    st.title("Система мониторинга и прогнозирования осложнений при бурении")
    st.markdown("---")

    # Контейнер для динамического обновления
    placeholder = st.empty()

    while True:
        with placeholder.container():
            # Получение новых данных
            live_df = data_gen.generate_live_data()

            # Преобразование литологии
            live_processed = pd.get_dummies(live_df, columns=['lithology'])

            # Прогнозирование
            try:
                features = ['depth', 'pressure', 'rop', 'vibration',
                            'lithology_limestone', 'lithology_sandstone', 'lithology_shale']
                live_processed['risk'] = model.predict_proba(live_processed[features])[:, 1]
            except Exception as e:
                st.error(f"Ошибка прогнозирования: {str(e)}")
                live_processed['risk'] = 0.0

            # Визуализация
            col1, col2 = st.columns([3, 1])

            with col1:
                # Графики
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=live_df['timestamp'],
                    y=live_df['pressure'],
                    name='Давление (psi)',
                    line=dict(color='#FF4B4B')
                ))
                fig.add_trace(go.Scatter(
                    x=live_df['timestamp'],
                    y=live_df['rop'],
                    name='Скорость бурения (m/h)',
                    yaxis='y2',
                    line=dict(color='#0068C9')
                ))
                fig.update_layout(
                    title='Параметры бурения в реальном времени',
                    yaxis=dict(title='Давление (psi)', color='#FF4B4B'),
                    yaxis2=dict(
                        title='ROP (m/h)',
                        overlaying='y',
                        side='right',
                        color='#0068C9'
                    ),
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

                # График риска
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Scatter(
                    x=live_df['timestamp'],
                    y=live_processed['risk'],
                    name='Риск осложнений',
                    line=dict(color='#FF9900', width=2)
                ))
                fig_risk.update_layout(
                    title='Прогноз риска осложнений',
                    yaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=300
                )
                st.plotly_chart(fig_risk, use_container_width=True)

                with col2:
                # Текущие показатели
                    current = live_processed.iloc[-1]
                st.metric("Текущая глубина", f"{current['depth']:.1f} м")
                st.metric("Давление", f"{current['pressure']:.1f} psi")
                st.metric("Скорость бурения", f"{current['rop']:.1f} m/h")
                st.metric("Уровень вибрации", f"{current['vibration']:.2f} g")

                # Индикатор риска
                risk_value = current['risk']
                risk_color = "#4CAF50" if risk_value < 0.3 else "#FFC107" if risk_value < 0.7 else "#F44336"
                st.markdown(f"""
                <div style="
                    background: {risk_color};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 10px 0;">
                    <h3 style="color: white; margin:0;">Текущий риск</h3>
                    <h1 style="color: white; margin:0;">{risk_value:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)

                # Рекомендации
                if risk_value > 0.7:
                    st.error("""
                    **Рекомендуемые действия:**
                    - Немедленно остановить бурение
                    - Проверить систему циркуляции
                    - Увеличить расход бурового раствора
                    """)
                elif risk_value > 0.4:
                    st.warning("""
                    **Рекомендуемые действия:**
                    - Увеличить мониторинг параметров
                    - Проверить показания датчиков
                    - Подготовить аварийный протокол
                    """)

                # Задержка для имитации реального времени
                time.sleep(2)

                if __name__ == "__main__":
                    main()
import streamlit as st
st.title("Мое Streamlit-приложение")







