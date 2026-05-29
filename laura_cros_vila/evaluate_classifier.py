# %%
import os
import numpy as np
import pickle
import pandas as pd
import json

from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_PATH = r"your/base_path"

def load_single_embedding(file):
    if not os.path.exists(file):
        print(f"[WARNING] Missing embedding: {file}")
        return None

    try:
        return np.load(file, mmap_mode="r")

    except Exception as e:
        print(f"[ERROR] Could not load {file}: {e}")
        return None


def load_embeddings(files):
    embeddings = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_single_embedding, f): f
            for f in files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading embeddings"
        ):
            emb = future.result()

            if emb is not None:
                embeddings.append(emb)

    return np.array(embeddings)


def get_split(split, embedding, folders):
    files = []
    y = []

    for folder in folders:

        emb_folder = os.path.join(
            BASE_PATH,
            folder,
            split,
            "embeddings",
            embedding
        )

        if not os.path.exists(emb_folder):
            print(f"[WARNING] Missing embeddings folder: {emb_folder}")
            continue

        for file_name in os.listdir(emb_folder):

            if file_name.endswith(".npy"):

                files.append(
                    os.path.join(emb_folder, file_name)
                )

                y.append(folder)

    if len(files) == 0:
        print("[ERROR] No embeddings found.")
        return np.array([]), np.array([]), []

    X = load_embeddings(files)
    y = np.array(y)

    return X, y, files


def load_ircamplify_results(folders):

    true_class = []
    files = []
    is_ai = []
    confidence = []

    ircamplify_path = os.path.join(
        BASE_PATH,
        "ircamplify_results"
    )

    for folder in folders:

        folder_path = os.path.join(
            ircamplify_path,
            folder
        )

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):

            if filename.endswith(".json"):

                with open(
                    os.path.join(folder_path, filename),
                    "r"
                ) as f:

                    data = json.load(f)

                job_infos = data.get("job_infos", {})
                file_paths = job_infos.get("file_paths", {})

                report_info = job_infos.get("report_info", {})
                report = report_info.get("report", {})

                result_list = report.get("resultList", [])

                for i, result in enumerate(result_list):

                    true_class.append(folder)

                    file = file_paths[i].split("/")[-1]

                    files.append(file)

                    is_ai.append(result.get("isAi"))

                    confidence.append(result.get("confidence"))

    return pd.DataFrame({
        "true_class": true_class,
        "file": files,
        "is_ai": is_ai,
        "confidence": confidence
    })

def get_classifiers_results(models, X_sample_scaled, sample_files):

    true_class = []
    files = []

    svm_pred = []
    rf_pred = []
    knn_pred = []

    for file in sample_files:

        normalized_path = os.path.normpath(file)
        path_parts = normalized_path.split(os.sep)

        # dataset structure:
        # BASE/folder/test/embeddings/model/file.npy
        folder_label = path_parts[-5]

        true_class.append(folder_label)

        files.append(
            os.path.basename(file).replace(".npy", ".mp3")
        )

    for name, model in models.items():

        y_pred = model.predict(X_sample_scaled)

        print(f"\n{name.upper()} unique predictions:")
        print(np.unique(y_pred))

        for pred in y_pred:

            if isinstance(pred, (list, np.ndarray)):
                pred = pred[0]

            pred_str = str(pred).lower()

            REAL_LABELS = ["nonai", "lastfm"]
            FAKE_LABELS = ["ai", "suno", "udio"]

            if pred_str in REAL_LABELS:
                binary_pred = "real"

            elif pred_str in FAKE_LABELS:
                binary_pred = "fake"

            else:
                binary_pred = "unknown"

            if name == "svc":
                svm_pred.append(binary_pred)

            elif name == "rf":
                rf_pred.append(binary_pred)

            elif name == "knn":
                knn_pred.append(binary_pred)

    return pd.DataFrame({
        "true_class": true_class,
        "file": files,

        "svm_pred": svm_pred,
        "rf_pred": rf_pred,
        "knn_pred": knn_pred
    })

def get_results_all(folders=["real", "fake"]):

    with open("models_and_scaler.pkl", "rb") as f:
        saved = pickle.load(f)

    models = saved["models"]
    scaler = saved["scaler"]

    X_sample, y_sample, sample_files = get_split(
        split="test",
        embedding="clap-laion-music",
        folders=folders
    )

    if len(X_sample) == 0:
        print("[ERROR] No embeddings loaded.")
        return None

    print(f"\nLoaded {len(X_sample)} embeddings")

    X_sample_scaled = scaler.transform(X_sample)

    classifiers_results = get_classifiers_results(
        models,
        X_sample_scaled,
        sample_files
    )

    ircamplify_path = os.path.join(
        BASE_PATH,
        "ircamplify_results"
    )

    if os.path.exists(ircamplify_path):

        ircamplify_results = load_ircamplify_results(
            folders
        ).drop_duplicates(
            subset="file",
            keep="first"
        )

        merged_data = pd.merge(
            classifiers_results,
            ircamplify_results,
            on=["true_class", "file"],
            how="left"
        )

        return merged_data

    else:
        return classifiers_results


def print_confusion_matrix_latex(y_true, y_pred, name):

    y_true_bool = np.array([
        label == "fake"
        for label in y_true
    ])

    y_pred_bool = np.array([
        label == "fake"
        for label in y_pred
    ])

    cm = pd.crosstab(
        y_true_bool,
        y_pred_bool,
        rownames=["True"],
        colnames=["Predicted"],
        normalize="index"
    )

    print(f"\n{name} confusion matrix:")
    print(cm.to_latex())


def print_classification_report_latex(data):

    y_true = data["true_class"]

    y_pred_svm = data["svm_pred"]
    y_pred_rf = data["rf_pred"]
    y_pred_knn = data["knn_pred"]

    print_confusion_matrix_latex(
        y_true,
        y_pred_svm,
        "SVM"
    )

    print_confusion_matrix_latex(
        y_true,
        y_pred_rf,
        "RF"
    )

    print_confusion_matrix_latex(
        y_true,
        y_pred_knn,
        "KNN"
    )

    if "is_ai" in data.columns:

        ircam_pred = np.array([
            "fake" if x else "real"
            for x in data["is_ai"]
        ])

        print_confusion_matrix_latex(
            y_true,
            ircam_pred,
            "Ircam Amplify"
        )

    table = (
        r"\begin{table}[ht]"
        r"\centering"
        r"\begin{tabular}{lcccc}"
        r"\hline"
        "\n"
    )

    table += (
        r"Classifier & Precision & Recall & F1-Score & Accuracy \\"
        r" \hline"
        "\n"
    )

    reports = []

    reports.append((
        "SVM",
        classification_report(
            y_true,
            y_pred_svm,
            output_dict=True,
            zero_division=0
        )
    ))

    reports.append((
        "RF",
        classification_report(
            y_true,
            y_pred_rf,
            output_dict=True,
            zero_division=0
        )
    ))

    reports.append((
        "KNN",
        classification_report(
            y_true,
            y_pred_knn,
            output_dict=True,
            zero_division=0
        )
    ))

    if "is_ai" in data.columns:

        reports.append((
            "Ircam Amp.",
            classification_report(
                y_true,
                ircam_pred,
                output_dict=True,
                zero_division=0
            )
        ))

    for name, report in reports:

        table += (
            f"{name} & "
            f"{report['macro avg']['precision']:.3f} & "
            f"{report['macro avg']['recall']:.3f} & "
            f"{report['macro avg']['f1-score']:.3f} & "
            f"{report['accuracy']:.3f} \\\\ \n"
        )

    table += (
        r"\hline"
        r"\end{tabular}"
        r"\caption{Binary classification results (Real vs Fake)}"
        r"\end{table}"
    )

    print("\n")
    print(table)


if __name__ == "__main__":

    folders = ["real", "fake"]

    data = get_results_all(folders)

    if data is not None:
        print_classification_report_latex(data)