# Titanic Classification API Service

MLOps API сервис для предсказания выживания пассажиров Титаника с использованием FastAPI.

## Возможности

- **FastAPI** с автоматической документацией Swagger
- **Поддержка нескольких моделей** (Random Forest Extended, Baseline)
- **Конфигурация через файлы** для выбора модели
- **Docker и Docker Compose** для развертывания различных версий
- **Prometheus метрики** для мониторинга
- **Grafana** для визуализации метрик
- **Скрипт бенчмарка** для сравнения производительности

## Быстрый старт

### 1. Локальный запуск

```bash
# Установка зависимостей
pip install -r service/requirements.txt

# Запуск API сервера
python run_api.py
```

API будет доступен по адресу: http://localhost:8000

- Документация: http://localhost:8000/docs
- Метрики: http://localhost:8000/metrics

### 2. Запуск с Docker Compose

```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Запуск в фоне
docker-compose up -d --build
```

Доступные сервисы:
- Random Forest Extended API: http://localhost:8001
- Baseline API: http://localhost:8002
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

## API Endpoints

### Основные endpoints

- `GET /` - Информация об API
- `GET /health` - Проверка статуса сервиса
- `GET /docs` - Swagger документация
- `GET /metrics` - Prometheus метрики

### Предсказания

- `POST /predict` - Предсказание для одного пассажира
- `POST /predict/batch` - Пакетное предсказание

### Управление моделями

- `GET /models/available` - Список доступных моделей
- `GET /model/info` - Информация о текущей модели
- `POST /model/load/{model_name}` - Загрузка модели

## Формат данных

### Входные данные пассажира

```json
{
  "Pclass": 3,
  "Sex": 1,
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked_C": 0,
  "Embarked_Q": 0
}
```

### Ответ предсказания

```json
{
  "prediction": 0,
  "probability": 0.23,
  "model_version": "model_extended.joblib"
}
```

## Конфигурация

### Переменные окружения

- `MODEL_TYPE` - Тип модели (random_forest, baseline)
- `DEFAULT_MODEL` - Файл модели по умолчанию
- `API_HOST` - Хост API (по умолчанию 0.0.0.0)
- `API_PORT` - Порт API (по умолчанию 8000)
- `LOG_LEVEL` - Уровень логирования

### Пример запуска с параметрами

```bash
MODEL_TYPE=baseline DEFAULT_MODEL=model_baseline.joblib python run_api.py
```

## Мониторинг

### Prometheus метрики

- `titanic_predictions_total` - Общее количество предсказаний
- `titanic_prediction_duration_seconds` - Время выполнения предсказаний
- `titanic_model_accuracy` - Точность модели
- `titanic_active_requests` - Количество активных запросов
- `titanic_errors_total` - Общее количество ошибок

### Grafana дашборды

После запуска docker-compose:
1. Откройте http://localhost:3000
2. Войдите с admin/admin123
3. Дашборды будут автоматически настроены

## Бенчмарк

Для сравнения производительности моделей используйте скрипт бенчмарка:

```bash
# Запустите API сервисы
docker-compose up -d

# Запустите бенчмарк
python benchmark_api.py
```

Результаты будут сохранены в `benchmark_results.json`.

## Структура проекта

```
service/
├── api/
│   ├── __init__.py
│   ├── app.py           # FastAPI приложение
│   ├── config.py        # Конфигурация
│   ├── schemas.py       # Pydantic схемы
│   ├── services.py      # Бизнес логика
│   ├── views.py         # API endpoints
│   └── metrics.py       # Prometheus метрики
├── requirements.txt     # Зависимости
├── Dockerfile          # Docker образ
└── README.md           # Документация
```

## Возможности расширения

1. **Добавление новых моделей**: Обновите `config.py` и добавьте новые конфигурации
2. **Новые метрики**: Расширьте `metrics.py`
3. **Дополнительные endpoints**: Добавьте в `views.py`
4. **Кастомные алерты**: Настройте правила в Prometheus

## Требования

- Python 3.12+
- Docker & Docker Compose
- FastAPI
- Polars
- Scikit-learn
- Prometheus-client
