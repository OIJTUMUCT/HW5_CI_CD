import os
import numpy as np
import pandas as pd

# ML / обучение
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# MLOps
import mlflow
import mlflow.sklearn
import joblib

# проверка данных
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

# анализ дрейфа
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


RANDOM_STATE = 42


def load_iris_df() -> pd.DataFrame:
    # Загрузка исходного датасета Iris и переименование целевой колонки в 'label'
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "label"})
    return df


def run_deepchecks(df: pd.DataFrame) -> str:
    # Анализ качества исходного датасета с помощью Deepchecks
    train_ds = Dataset(
        df.drop(columns=["label"]),
        label=df["label"],
        cat_features=[],  # все признаки Iris — числовые
    )

    suite = data_integrity()
    result = suite.run(train_ds)

    os.makedirs("reports", exist_ok=True)
    dc_report_path = os.path.join("reports", "deepchecks_report.html")
    result.save_as_html(dc_report_path)

    print("Data Integrity Suite:")
    print(f"Deepchecks report saved to {dc_report_path}")
    return dc_report_path


def evidently_analysis(df: pd.DataFrame) -> str:
    # Анализ дрейфа данных с EvidentlyAI (70/30).
    reference_data = df.sample(frac=0.7, random_state=RANDOM_STATE)
    current_data = df.drop(reference_data.index)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    os.makedirs("reports", exist_ok=True)
    output_path = "reports/evidently_report.html"
    report.save_html(output_path)

    print(f"EvidentlyAI отчёт сохранён в {output_path}")
    return output_path


def mlflow_experiment(df: pd.DataFrame):
    """Запуск MLflow-эксперимента: RF на Iris, логирование метрик и модели."""
    np.random.seed(RANDOM_STATE)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # настройка MLflow с проверкой директорий
    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri_env:
        mlflow.set_tracking_uri(tracking_uri_env)
    else:
        mlruns_root = os.path.abspath("./mlruns")
        trash_dir = os.path.join(mlruns_root, ".trash")

        if os.path.exists(mlruns_root) and not os.path.isdir(mlruns_root):
            raise RuntimeError(
                f"Путь {mlruns_root} существует и не является директорией. "
                f"Удалите/переименуйте этот файл."
            )

        os.makedirs(mlruns_root, exist_ok=True)
        os.makedirs(trash_dir, exist_ok=True)

        mlflow.set_tracking_uri(f"file:{mlruns_root}")

    mlflow.set_experiment("iris_hw5")

    with mlflow.start_run(run_name="rf_iris_baseline"):
        params = {
            "n_estimators": 100,
            "max_depth": 12,
            "random_state": RANDOM_STATE,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # логирование гиперпараметров и метрик
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # пример входных данных для UI
        input_example = X_test.iloc[:5]

        # логирование модели в MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

        # локальный артефакт модели
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)

        # логирование pkl
        mlflow.log_artifact(model_path)

        print(f"Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
        print("Run id:", mlflow.active_run().info.run_id)

    return model, acc, f1


if __name__ == "__main__":
    # загрузка данных
    df = load_iris_df()

    # Deepchecks
    run_deepchecks(df)

    # EvidentlyAI
    evidently_analysis(df)

    # MLflow-эксперимент
    model, acc, f1 = mlflow_experiment(df)
    print("Pipeline finished.")