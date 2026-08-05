# ============================================================
# Smart EDA Dashboard — Machine Learning Module
# Prediction-focused ML: preprocessing, pipelines, CV, tuning,
# classification, regression, clustering/PCA, and MLP neural nets
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from modules.models.data_audit import (
    build_model_data_audit,
    render_model_data_audit,
    render_train_test_audit,
)

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    GroupKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance


# ============================================================
# Generic helpers
# ============================================================
def _get_working_df(df, df_cleaned):
    return df_cleaned if df_cleaned is not None else df


def _make_ohe():
    """Compatibility wrapper for sklearn versions with sparse/sparse_output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_ordinal():
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _to_text_1d(x):
    arr = np.asarray(x).ravel()
    return pd.Series(arr).fillna("").astype(str)


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def _safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)


def _infer_task_type(df, target_col):
    if target_col is None or target_col == "No target — clustering / PCA":
        return "Clustering / Unsupervised"

    y = df[target_col].dropna()
    nunique = y.nunique()

    if nunique <= 1:
        return "Not enough target variation"
    if not _is_numeric_series(y):
        return "Classification"
    if nunique <= 10:
        return "Classification"
    return "Regression"


def _prepare_xy(data, target_col, predictor_cols):
    cols = predictor_cols + [target_col]
    model_df = data[cols].copy()
    model_df = model_df.dropna(subset=[target_col])
    audit = build_model_data_audit(
        data,
        model_df,
        cols,
        "Rows with a missing target are excluded; predictor missingness is handled inside the training Pipeline.",
    )
    model_df.attrs["model_data_audit"] = audit.to_dict()
    X = model_df[predictor_cols]
    y = model_df[target_col]
    return model_df, X, y


def _detect_text_candidates(data, predictor_cols):
    candidates = []
    for c in predictor_cols:
        if c not in data.columns:
            continue
        s = data[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(200)
            if len(sample) == 0:
                continue
            avg_words = sample.str.split().str.len().mean()
            avg_len = sample.str.len().mean()
            unique_ratio = sample.nunique() / max(len(sample), 1)
            if avg_words >= 3 or avg_len >= 25 or unique_ratio > 0.70:
                candidates.append(c)
    return candidates


def _build_preprocessor(
    X,
    scale_numeric=True,
    categorical_encoding="OneHotEncoder",
    text_features=None,
    tfidf_max_features=500,
):
    """
    Builds a leakage-safe ColumnTransformer:
    - numeric: SimpleImputer + optional StandardScaler
    - categorical: SimpleImputer + OneHotEncoder or OrdinalEncoder
    - text: TF-IDF per selected text column
    """
    text_features = [c for c in (text_features or []) if c in X.columns]
    numeric_features = [c for c in X.select_dtypes(include=[np.number]).columns.tolist() if c not in text_features]
    categorical_features = [c for c in X.columns if c not in numeric_features and c not in text_features]

    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    if categorical_encoding == "OrdinalEncoder":
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", _make_ordinal()),
            ]
        )
    else:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_ohe()),
            ]
        )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    for col in text_features:
        text_pipe = Pipeline(
            steps=[
                ("to_text", FunctionTransformer(_to_text_1d, validate=False)),
                ("tfidf", TfidfVectorizer(max_features=int(tfidf_max_features), ngram_range=(1, 1))),
            ]
        )
        transformers.append((f"tfidf_{col}", text_pipe, col))

    return (
        ColumnTransformer(transformers=transformers, remainder="drop"),
        numeric_features,
        categorical_features,
        text_features,
    )


def _get_feature_names(preprocessor, numeric_features, categorical_features, text_features):
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        names.extend(numeric_features)
        names.extend(categorical_features)
        for col in text_features:
            try:
                tfidf = preprocessor.named_transformers_[f"tfidf_{col}"].named_steps["tfidf"]
                names.extend([f"{col}:{w}" for w in tfidf.get_feature_names_out().tolist()])
            except Exception:
                names.append(col)
        return names


def _binary_specificity(y_true, y_pred, positive_label):
    labels = list(pd.Series(y_true).dropna().unique())
    if len(labels) != 2:
        return np.nan
    neg_label = [l for l in labels if l != positive_label][0]
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, positive_label])
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def _plot_confusion_matrix(y_true, y_pred, labels, title, plot_template):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[str(x) for x in labels], columns=[str(x) for x in labels])
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=title,
        color_continuous_scale="Blues",
    )
    fig.update_layout(template=plot_template)
    return fig


def _extract_importance(fitted_pipeline, feature_names, model_name):
    try:
        model = fitted_pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            vals = np.mean(np.abs(coef), axis=0) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
        else:
            return None
        n = min(len(feature_names), len(vals))
        imp = pd.DataFrame({"Feature": feature_names[:n], "Importance": vals[:n]})
        imp = imp.sort_values("Importance", ascending=False).head(25)
        imp["Model"] = model_name
        return imp
    except Exception:
        return None



def _direct_importance_method(model_name):
    """Return the direct importance method that fits the selected model family."""
    if any(m in model_name for m in ["Decision Tree", "Random Forest", "Gradient Boosting"]):
        return "Tree impurity / split-based feature importance"
    if any(m in model_name for m in ["Logistic Regression", "Linear Regression", "Ridge", "Lasso", "Elastic Net"]):
        return "Absolute coefficient size"
    return None


def _supports_shap_fast(model_name):
    """Keep SHAP limited to model families where it is usually useful and not too slow."""
    supported = [
        "Decision Tree", "Random Forest", "Gradient Boosting",
        "Logistic Regression", "Linear Regression", "Ridge", "Lasso", "Elastic Net",
    ]
    return any(m in model_name for m in supported)


def _explainability_recommendation(model_name):
    direct = _direct_importance_method(model_name)
    if direct:
        return (
            f"For {model_name}, direct importance is available via {direct}. "
            "Permutation importance can confirm whether those features improve test-set prediction. "
            "SHAP is useful here for model explanation if the package is installed."
        )
    if any(m in model_name for m in ["KNN", "SVM", "Neural Network", "Naive Bayes"]):
        return (
            f"For {model_name}, direct feature importance is not reliable or not available. "
            "Use permutation importance to estimate which original variables affect prediction. "
            "SHAP is not enabled by default for this model family because model-agnostic SHAP can be slow and unstable on large data."
        )
    return "Use permutation importance as the general explanation method."

def _needs_dense(model_name):
    dense_models = ["Naive Bayes", "Neural Network"]
    return any(m in model_name for m in dense_models)


def _make_cv(cv_choice, n_splits, task):
    n_splits = int(n_splits)
    if cv_choice == "KFold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    if cv_choice == "StratifiedKFold" and task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    if cv_choice == "GroupKFold":
        return GroupKFold(n_splits=n_splits)
    return None


def _classification_param_grid(model_name):
    grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
        "KNN": {"model__n_neighbors": [3, 5, 9]},
        "Decision Tree": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 5, 10],
            "model__ccp_alpha": [0.0, 0.001],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "SVM": {"model__C": [0.5, 1.0, 2.0], "model__kernel": ["rbf", "linear"]},
        "Neural Network (MLP)": {
            "model__alpha": [0.0001, 0.001],
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__activation": ["relu", "tanh"],
            "model__learning_rate_init": [0.001, 0.01],
        },
    }
    return grids.get(model_name, {})


def _regression_param_grid(model_name):
    grids = {
        "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
        "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        "Elastic Net": {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
        "KNN": {"model__n_neighbors": [3, 5, 9]},
        "Decision Tree": {
            "model__criterion": ["squared_error", "absolute_error"],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 5, 10],
            "model__ccp_alpha": [0.0, 0.001],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "SVM": {"model__C": [0.5, 1.0, 2.0], "model__kernel": ["rbf", "linear"]},
        "Neural Network (MLP)": {
            "model__alpha": [0.0001, 0.001],
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__activation": ["relu", "tanh"],
            "model__learning_rate_init": [0.001, 0.01],
        },
    }
    return grids.get(model_name, {})


def _simple_note_for_classification(best_row, imbalanced=False):
    model = best_row["Model"]
    auc = best_row.get("ROC-AUC", np.nan)
    f1 = best_row.get("F1", np.nan)
    metric_hint = "PR-AUC, recall, and F1" if imbalanced else "ROC-AUC, F1, and calibration"
    if pd.notna(auc):
        return f"Best model by ROC-AUC is {model} (AUC={auc:.3f}). Recommended focus: {metric_hint}."
    return f"Best model by F1 is {model} (F1={f1:.3f}). Recommended focus: {metric_hint}."


def _simple_note_for_regression(best_row):
    return f"Best model by lowest RMSE is {best_row['Model']} (RMSE={best_row['RMSE']:.3f}, MAE={best_row['MAE']:.3f})."


# ============================================================
# Practical recommendation blocks
# ============================================================
def _render_model_learning_guide(selected_models, scale_numeric=True, context="classification"):
    """Compact practical recommendations shown only for selected model families."""
    selected_text = " | ".join(selected_models)

    with st.expander("Practical recommendations based on selected models", expanded=False):
        if "Decision Tree" in selected_text:
            st.markdown(
                """
                **Decision Tree**
                - Easy to understand and can be visualized as step-by-step decisions.
                - Captures non-linear relationships and gives feature importance.
                - Usually does **not** need StandardScaler.
                - Main risk: **overfitting**, especially when the tree is too deep.
                - Important concepts: Root node, internal node, leaf node, depth, pruning, feature importance.
                - Key hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `ccp_alpha`.
                """
            )
            st.info("Recommendation: start with a limited depth such as max_depth=3 or 5, then use GridSearchCV to tune depth and pruning.")

        if "Random Forest" in selected_text:
            st.markdown(
                """
                **Random Forest**
                - An ensemble of many decision trees using **bagging**.
                - Each tree is trained on a different sample, which makes the model more stable.
                - Usually reduces overfitting compared with a single Decision Tree.
                - Usually gives higher accuracy and useful feature importance.
                - Important hyperparameter: `n_estimators` = number of trees.
                """
            )

        if "Gradient Boosting" in selected_text:
            st.markdown(
                """
                **Gradient Boosting**
                - Builds trees sequentially; each new tree tries to correct previous errors.
                - Often strong predictive performance.
                - More sensitive to hyperparameters than Random Forest.
                - Key hyperparameters: `learning_rate`, `n_estimators`, `max_depth`.
                """
            )
            st.warning("Recommendation: tune learning_rate and n_estimators. Smaller learning_rate often needs more trees.")

        needs_scaling_models = [
            "Linear Regression", "Logistic Regression", "Ridge", "Lasso", "Elastic Net",
            "KNN", "SVM", "Neural Network"
        ]
        tree_models = ["Decision Tree", "Random Forest", "Gradient Boosting"]
        if any(m in selected_text for m in needs_scaling_models):
            if scale_numeric:
                st.success("Scaling recommendation: StandardScaler is ON, which is appropriate for Linear/Logistic models, KNN, SVM, and Neural Networks.")
            else:
                st.warning("Scaling recommendation: turn StandardScaler ON if you use Linear/Logistic models, KNN, SVM, or Neural Networks.")
        if any(m in selected_text for m in tree_models):
            st.caption("Tree-based models usually do not require scaling, but keeping scaling inside the Pipeline is still safe for mixed model comparison.")

        if any(m in selected_text for m in ["Ridge", "Lasso", "Elastic Net", "Logistic Regression", "SVM"]):
            st.markdown(
                """
                **Regularization**
                - Important for linear models, Logistic Regression, and SVM.
                - **Ridge** shrinks coefficients.
                - **Lasso** can shrink some coefficients to zero, which helps feature selection.
                - **Elastic Net** combines Ridge and Lasso.
                - In Logistic Regression and SVM, smaller `C` means stronger regularization.
                """
            )

        if "Neural Network" in selected_text:
            st.markdown(
                """
                **Neural Network (MLP)**
                - Use when the goal is prediction and the data may contain complex non-linear patterns.
                - It usually needs: missing-value handling, categorical encoding, and **StandardScaler** for numeric predictors.
                - It can overfit on small datasets, so keep **early stopping** on and compare against simpler models.
                - Key settings: hidden layer size, activation function, regularization `alpha`, batch size, learning rate, and maximum iterations.
                - `ReLU` is a strong default hidden-layer activation; `tanh` can be useful in some smaller scaled datasets.
                """
            )
            if scale_numeric:
                st.success("Neural network recommendation: StandardScaler is ON and early stopping is available, which helps MLP training stability.")
            else:
                st.warning("Neural network recommendation: turn StandardScaler ON before using MLP.")

        st.markdown(
            """
            **GridSearchCV** tests multiple hyperparameter values using cross-validation on the training set only.
            This helps select better model settings while reducing data leakage when used inside a Pipeline.
            """
        )


def _render_short_model_comparison():
    with st.expander("Model choice notes: tree-based models", expanded=False):
        comp = pd.DataFrame([
            {"Model": "Decision Tree", "Speed": "Very fast", "Interpretability": "High", "Overfitting risk": "High", "Typical accuracy": "Moderate"},
            {"Model": "Random Forest", "Speed": "Medium", "Interpretability": "Medium", "Overfitting risk": "Lower", "Typical accuracy": "High"},
            {"Model": "Gradient Boosting", "Speed": "Slower", "Interpretability": "Medium/Low", "Overfitting risk": "Medium if tuned", "Typical accuracy": "Often highest"},
        ])
        st.dataframe(comp, use_container_width=True)
        st.caption("Depth ↑ usually increases overfitting risk. Pruning or limiting max_depth helps improve generalization.")


# ============================================================
# Shared preprocessing UI
# ============================================================
def _preprocessing_options(data, predictor_cols, prefix):
    st.markdown("### Preprocessing inside Pipeline")
    st.caption(
        "SimpleImputer, encoding, scaling, TF-IDF, and model fitting are inside one Pipeline to reduce data leakage."
    )

    text_candidates = _detect_text_candidates(data, predictor_cols)
    p1, p2, p3 = st.columns(3)
    categorical_encoding = p1.selectbox(
        "Categorical encoding",
        ["OneHotEncoder", "OrdinalEncoder"],
        index=0,
        key=f"{prefix}_cat_encoding",
        help="OneHotEncoder is safer for nominal categories. OrdinalEncoder is useful for ordered categories or tree models, but can imply artificial order.",
    )
    scale_numeric = p2.checkbox(
        "StandardScaler for numeric predictors",
        value=True,
        key=f"{prefix}_scale",
        help="Important for Logistic Regression, SVM, KNN, and Neural Networks. Less important for trees.",
    )
    tfidf_max_features = p3.slider(
        "TF-IDF max features",
        50,
        2000,
        500,
        50,
        key=f"{prefix}_tfidf_max",
        help="Used only for selected text columns.",
    )

    text_features = st.multiselect(
        "Text columns for TF-IDF (optional)",
        [c for c in predictor_cols if c in data.columns],
        default=text_candidates[: min(2, len(text_candidates))],
        key=f"{prefix}_text_cols",
        help="Use for free-text columns. Do not select ordinary categorical columns here.",
    )

    return categorical_encoding, scale_numeric, text_features, tfidf_max_features


# ============================================================
# Classification
# ============================================================
def _render_classification(data, target_col, predictor_cols, plot_template):
    st.markdown("### Classification — prediction models")
    st.caption("Use this when the target is binary or categorical. The goal is prediction performance on unseen data, not p-values.")

    with st.expander("Recommended workflow", expanded=True):
        st.markdown(
            """
            **EDA → Select X/Y → Train/Test split → Preprocessing → ColumnTransformer → Pipeline → Model → GridSearchCV/CV → Fit → Predict → Evaluate → Threshold tuning → Final model selection**

            The app keeps preprocessing inside the Pipeline, so imputation/encoding/scaling are learned from the training data only.
            """
        )

    model_df, X, y = _prepare_xy(data, target_col, predictor_cols)
    render_model_data_audit(model_df.attrs["model_data_audit"])

    if y.nunique() < 2:
        st.error("The target must have at least two classes.")
        return

    classes = sorted(y.dropna().unique().tolist(), key=lambda x: str(x))
    imbalanced = y.value_counts(normalize=True).min() < 0.20

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size", 0.10, 0.50, 0.20, 0.05, key="ml_cls_test")
    random_state = int(c2.number_input("Random seed", value=42, step=1, key="ml_cls_seed"))
    stratify_default = True if imbalanced or len(classes) == 2 else False
    stratify_on = c3.checkbox("Stratified split", value=stratify_default, key="ml_cls_stratify")

    if imbalanced:
        st.warning("Class imbalance detected. Use Stratified Split and focus on PR-AUC, Recall, Precision, F1, and threshold tuning — not accuracy only.")
    else:
        st.info("Classes look reasonably balanced. ROC-AUC is useful, but still check confusion matrix, recall, precision, and F1.")

    positive_label = None
    if len(classes) == 2:
        positive_label = st.selectbox("Positive class", classes, index=len(classes) - 1, key="ml_positive_class")
        st.caption("Binary classification: focus on the positive class, threshold tuning, recall, precision, F1, ROC-AUC, and PR-AUC.")
    else:
        st.caption("Multi-class classification: probability-based models output one probability per class; choose the class with the highest probability. Use weighted F1 / ROC-AUC and inspect the confusion matrix.")

    categorical_encoding, scale_numeric, text_features, tfidf_max_features = _preprocessing_options(
        data=model_df,
        predictor_cols=predictor_cols,
        prefix="ml_cls",
    )

    selected_models = st.multiselect(
        "Choose models",
        [
            "Logistic Regression (Prediction Baseline)",
            "KNN Classifier",
            "Naive Bayes",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
            "SVM Classifier",
            "Neural Network (MLP) — Advanced",
        ],
        default=[
            "Logistic Regression (Prediction Baseline)",
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
        ],
        key="ml_cls_models",
    )

    _render_model_learning_guide(selected_models, scale_numeric=scale_numeric, context="classification")
    _render_short_model_comparison()

    with st.expander("Advanced settings and practical recommendations"):
        st.markdown(
            """
            - **Logistic Regression** uses L2 regularization by default. Smaller `C` = stronger regularization.
            - **SVM** also uses regularization through `C`.
            - **Neural Network (MLP)** is advanced; use it mainly for prediction, not p-values or simple interpretation.
            - For MLP, keep preprocessing inside the Pipeline: imputation, encoding, and StandardScaler.
            - Early stopping uses a validation split from the training data to reduce overfitting.
            """
        )
        nn1, nn2, nn3 = st.columns(3)
        hidden_units = nn1.slider("MLP hidden layer units", 10, 200, 50, 10, key="ml_cls_hidden")
        max_iter = nn2.slider("MLP max iterations", 100, 1000, 500, 50, key="ml_cls_iter", help="Similar idea to epochs: more iterations means longer training, but can overfit.")
        activation = nn3.selectbox("MLP activation", ["relu", "tanh", "logistic"], key="ml_cls_activation", help="ReLU is a strong default. Logistic is sigmoid-like and can train slowly.")
        nn4, nn5, nn6 = st.columns(3)
        batch_size = nn4.selectbox("Batch size", [16, 32, 64, 128], index=1, key="ml_cls_batch")
        learning_rate_init = nn5.selectbox("Learning rate", [0.0005, 0.001, 0.005, 0.01], index=1, key="ml_cls_lr")
        early_stopping = nn6.checkbox("Early stopping", value=True, key="ml_cls_early")

    with st.expander("Cross-validation and hyperparameter tuning"):
        enable_tuning = st.checkbox("Use GridSearchCV", value=False, key="ml_cls_grid")
        cv_choice = st.selectbox("CV method", ["StratifiedKFold", "KFold", "GroupKFold"], key="ml_cls_cv")
        n_splits = st.slider("Number of folds", 3, 10, 5, key="ml_cls_folds")
        group_col = None
        if cv_choice == "GroupKFold":
            group_col = st.selectbox("Group column", [c for c in data.columns if c != target_col], key="ml_cls_group")
            st.caption("GroupKFold keeps observations from the same group together in train/test folds.")
        st.caption("For large datasets or many models, GridSearchCV can take longer.")

    if not selected_models:
        st.warning("Select at least one model.")
        return

    stratify = y if stratify_on and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        render_train_test_audit(len(model_df), len(X_train), len(X_test))
    except Exception as e:
        st.error("Could not split the data. Try disabling stratification or using a larger dataset.")
        st.code(str(e))
        return

    preprocessor, numeric_features, categorical_features, text_features = _build_preprocessor(
        X,
        scale_numeric=scale_numeric,
        categorical_encoding=categorical_encoding,
        text_features=text_features,
        tfidf_max_features=tfidf_max_features,
    )

    models = {}
    if "Logistic Regression (Prediction Baseline)" in selected_models:
        models["Logistic Regression"] = LogisticRegression(max_iter=1500, class_weight="balanced")
    if "KNN Classifier" in selected_models:
        models["KNN"] = KNeighborsClassifier()
    if "Naive Bayes" in selected_models:
        models["Naive Bayes"] = GaussianNB()
    if "Decision Tree Classifier" in selected_models:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    if "Random Forest Classifier" in selected_models:
        models["Random Forest"] = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    if "Gradient Boosting Classifier" in selected_models:
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=random_state)
    if "SVM Classifier" in selected_models:
        models["SVM"] = SVC(probability=True, random_state=random_state, class_weight="balanced")
    if "Neural Network (MLP) — Advanced" in selected_models:
        models["Neural Network (MLP)"] = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation=activation,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    results = []
    fitted = {}
    predictions_store = pd.DataFrame(index=X_test.index)
    predictions_store["Actual"] = y_test

    cv = _make_cv(cv_choice, n_splits, task="classification") if enable_tuning else None
    groups = model_df.loc[X_train.index, group_col] if enable_tuning and cv_choice == "GroupKFold" and group_col else None
    scoring = "roc_auc" if len(classes) == 2 else "f1_weighted"

    progress = st.progress(0)
    for i, (name, model) in enumerate(models.items(), start=1):
        try:
            steps = [("preprocess", preprocessor)]
            if _needs_dense(name):
                steps.append(("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)))
            steps.append(("model", model))
            pipe = Pipeline(steps=steps)

            best_params = "Not tuned"
            if enable_tuning:
                param_grid = _classification_param_grid(name)
                if param_grid:
                    search = GridSearchCV(
                        pipe,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                    )
                    search.fit(X_train, y_train, groups=groups)
                    pipe = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipe.fit(X_train, y_train)
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            avg = "binary" if len(classes) == 2 else "weighted"
            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average=avg, pos_label=positive_label, zero_division=0),
                "Recall / Sensitivity": recall_score(y_test, y_pred, average=avg, pos_label=positive_label, zero_division=0),
                "F1": f1_score(y_test, y_pred, average=avg, pos_label=positive_label, zero_division=0),
                "Best Params": str(best_params),
            }

            if len(classes) == 2:
                row["Specificity"] = _binary_specificity(y_test, y_pred, positive_label)

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)
                if len(classes) == 2:
                    class_list = list(pipe.named_steps["model"].classes_)
                    pos_idx = class_list.index(positive_label)
                    y_score = proba[:, pos_idx]
                    y_binary = (y_test == positive_label).astype(int)
                    row["ROC-AUC"] = roc_auc_score(y_binary, y_score)
                    row["PR-AUC"] = average_precision_score(y_binary, y_score)
                    row["Brier Score"] = brier_score_loss(y_binary, y_score)
                    predictions_store[f"{name} probability"] = y_score
                else:
                    row["ROC-AUC"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                    row["PR-AUC"] = np.nan
                    row["Brier Score"] = np.nan
            else:
                row["ROC-AUC"] = np.nan
                row["PR-AUC"] = np.nan
                row["Brier Score"] = np.nan

            predictions_store[f"{name} predicted"] = y_pred
            results.append(row)
            fitted[name] = pipe
        except Exception as e:
            st.warning(f"{name} could not run: {e}")
        progress.progress(i / max(len(models), 1))
    progress.empty()

    if not results:
        st.error("No model could be fitted.")
        return

    results_df = pd.DataFrame(results)
    display_df = results_df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)

    st.markdown("### Model comparison")
    st.caption("For classification: higher ROC-AUC/PR-AUC/F1 is better. Lower Brier Score is better because it measures probability calibration error.")
    st.dataframe(display_df, use_container_width=True)

    if imbalanced and "PR-AUC" in results_df.columns and results_df["PR-AUC"].notna().any():
        metric_for_best = "PR-AUC"
    else:
        metric_for_best = "ROC-AUC" if results_df["ROC-AUC"].notna().any() else "F1"
    best_idx = results_df[metric_for_best].astype(float).idxmax()
    best_row = results_df.loc[best_idx]
    best_model_name = best_row["Model"]
    st.success(_simple_note_for_classification(best_row, imbalanced=imbalanced))

    best_pipe = fitted[best_model_name]
    best_pred = best_pipe.predict(X_test)
    st.markdown("### Confusion matrix — best model")
    st.plotly_chart(
        _plot_confusion_matrix(y_test, best_pred, classes, f"Confusion Matrix: {best_model_name}", plot_template),
        use_container_width=True,
    )

    if len(classes) == 2 and hasattr(best_pipe, "predict_proba"):
        class_list = list(best_pipe.named_steps["model"].classes_)
        pos_idx = class_list.index(positive_label)
        y_score = best_pipe.predict_proba(X_test)[:, pos_idx]
        y_binary = (y_test == positive_label).astype(int)

        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_df = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
        fig = px.line(roc_df, x="False positive rate", y="True positive rate", title=f"ROC Curve: {best_model_name}")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        fig.update_layout(template=plot_template)
        st.plotly_chart(fig, use_container_width=True)

        prec, rec, thresholds = precision_recall_curve(y_binary, y_score)
        pr_df = pd.DataFrame({"Recall": rec, "Precision": prec})
        fig_pr = px.line(pr_df, x="Recall", y="Precision", title=f"Precision-Recall Curve: {best_model_name}")
        fig_pr.update_layout(template=plot_template)
        st.plotly_chart(fig_pr, use_container_width=True)

        try:
            brier = brier_score_loss(y_binary, y_score)
            st.metric("Brier Score — probability calibration", f"{brier:.4f}")
            st.caption("Brier Score evaluates how close predicted probabilities are to the real outcome. Lower is better; 0 is perfect.")
        except Exception:
            pass

        st.markdown("### Threshold tuning")
        st.caption("Default threshold is 0.5. Lower threshold → Recall increases and Precision usually decreases. Higher threshold → Precision increases and Recall usually decreases.")
        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05, key="ml_cls_threshold")
        tuned_pred = np.where(y_score >= threshold, positive_label, [c for c in classes if c != positive_label][0])
        tuned_metrics = pd.DataFrame([
            {
                "Threshold": threshold,
                "Precision": precision_score(y_test, tuned_pred, pos_label=positive_label, zero_division=0),
                "Recall / Sensitivity": recall_score(y_test, tuned_pred, pos_label=positive_label, zero_division=0),
                "Specificity": _binary_specificity(y_test, tuned_pred, positive_label),
                "F1": f1_score(y_test, tuned_pred, pos_label=positive_label, zero_division=0),
            }
        ]).round(4)
        st.dataframe(tuned_metrics, use_container_width=True)
        st.plotly_chart(
            _plot_confusion_matrix(y_test, tuned_pred, classes, f"Confusion Matrix at threshold {threshold:.2f}", plot_template),
            use_container_width=True,
        )

    st.markdown("### Feature importance / coefficient importance")
    try:
        feature_names = _get_feature_names(
            best_pipe.named_steps["preprocess"],
            numeric_features,
            categorical_features,
            text_features,
        )
        imp = _extract_importance(best_pipe, feature_names, best_model_name)
        if imp is not None and not imp.empty:
            fig_imp = px.bar(
                imp.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top features: {best_model_name}",
            )
            fig_imp.update_layout(template=plot_template, height=650)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(imp.round(5), use_container_width=True)
        else:
            st.info("This model does not provide direct feature importance. Use Random Forest/Gradient Boosting or add permutation importance later.")
    except Exception:
        st.info("Feature importance could not be calculated for this model.")

    try:
        _render_advanced_explainability(best_pipe, X_test, y_test, "classification", feature_names, plot_template)
    except Exception:
        pass

    st.markdown("### Download predictions")
    csv = predictions_store.reset_index(drop=False).to_csv(index=False).encode("utf-8")
    st.download_button("Download test-set predictions", csv, file_name="ml_classification_predictions.csv", mime="text/csv")


# ============================================================
# Regression
# ============================================================
def _render_regression(data, target_col, predictor_cols, plot_template):
    st.markdown("### Regression — numeric prediction models")
    st.caption("Use this when the target is continuous, such as cost, length of stay, score, or blood pressure.")

    with st.expander("Recommended workflow", expanded=True):
        st.markdown(
            """
            **EDA → Select X/Y → Train/Test split → Preprocessing → ColumnTransformer → Pipeline → Model → Param Grid → GridSearchCV/CV → Fit → Predict → Evaluate → Final model selection**

            Regression metrics: **RMSE**, **MAE**, **R²**. Lower RMSE/MAE is better; higher R² is better.
            """
        )

    model_df, X, y = _prepare_xy(data, target_col, predictor_cols)
    render_model_data_audit(model_df.attrs["model_data_audit"])
    y = pd.to_numeric(y, errors="coerce")
    keep = y.notna()
    X = X.loc[keep]
    y = y.loc[keep]
    model_df = model_df.loc[keep]

    if y.nunique() < 2:
        st.error("The target must have variation.")
        return

    c1, c2 = st.columns(2)
    test_size = c1.slider("Test size", 0.10, 0.50, 0.20, 0.05, key="ml_reg_test")
    random_state = int(c2.number_input("Random seed", value=42, step=1, key="ml_reg_seed"))

    categorical_encoding, scale_numeric, text_features, tfidf_max_features = _preprocessing_options(
        data=model_df,
        predictor_cols=predictor_cols,
        prefix="ml_reg",
    )

    selected_models = st.multiselect(
        "Choose models",
        [
            "Linear Regression (Prediction Baseline)",
            "Ridge Regression",
            "Lasso Regression",
            "Elastic Net Regression",
            "KNN Regressor",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "SVM Regressor",
            "Neural Network Regressor (MLP) — Advanced",
        ],
        default=[
            "Linear Regression (Prediction Baseline)",
            "Ridge Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
        ],
        key="ml_reg_models",
    )

    _render_model_learning_guide(selected_models, scale_numeric=scale_numeric, context="regression")
    _render_short_model_comparison()

    with st.expander("Advanced settings and practical recommendations"):
        st.markdown(
            """
            - **Ridge** shrinks coefficients and is useful with correlated predictors.
            - **Lasso** can shrink some coefficients to zero, helping feature selection.
            - **Elastic Net** combines Ridge and Lasso.
            - **Neural Network (MLP)** is advanced and works better with larger datasets and scaled numeric features.
            - For MLP, use early stopping and compare against Random Forest / Gradient Boosting before choosing it as the final model.
            """
        )
        nn1, nn2, nn3 = st.columns(3)
        hidden_units = nn1.slider("MLP hidden layer units", 10, 200, 50, 10, key="ml_reg_hidden")
        max_iter = nn2.slider("MLP max iterations", 100, 1000, 500, 50, key="ml_reg_iter", help="Similar idea to epochs: more iterations means longer training, but can overfit.")
        activation = nn3.selectbox("MLP activation", ["relu", "tanh", "logistic"], key="ml_reg_activation", help="ReLU is a strong default. Tanh can work well with scaled data.")
        nn4, nn5, nn6 = st.columns(3)
        batch_size = nn4.selectbox("Batch size", [16, 32, 64, 128], index=1, key="ml_reg_batch")
        learning_rate_init = nn5.selectbox("Learning rate", [0.0005, 0.001, 0.005, 0.01], index=1, key="ml_reg_lr")
        early_stopping = nn6.checkbox("Early stopping", value=True, key="ml_reg_early")

    with st.expander("Cross-validation and hyperparameter tuning"):
        enable_tuning = st.checkbox("Use GridSearchCV", value=False, key="ml_reg_grid")
        cv_choice = st.selectbox("CV method", ["KFold", "GroupKFold"], key="ml_reg_cv")
        n_splits = st.slider("Number of folds", 3, 10, 5, key="ml_reg_folds")
        group_col = None
        if cv_choice == "GroupKFold":
            group_col = st.selectbox("Group column", [c for c in data.columns if c != target_col], key="ml_reg_group")
        st.caption("GridSearchCV chooses hyperparameters using cross-validation on the training set only.")

    if not selected_models:
        st.warning("Select at least one model.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        render_train_test_audit(len(model_df), len(X_train), len(X_test))
    except Exception as e:
        st.error("Could not split the data.")
        st.code(str(e))
        return

    preprocessor, numeric_features, categorical_features, text_features = _build_preprocessor(
        X,
        scale_numeric=scale_numeric,
        categorical_encoding=categorical_encoding,
        text_features=text_features,
        tfidf_max_features=tfidf_max_features,
    )

    models = {}
    if "Linear Regression (Prediction Baseline)" in selected_models:
        models["Linear Regression"] = LinearRegression()
    if "Ridge Regression" in selected_models:
        models["Ridge"] = Ridge()
    if "Lasso Regression" in selected_models:
        models["Lasso"] = Lasso(max_iter=5000)
    if "Elastic Net Regression" in selected_models:
        models["Elastic Net"] = ElasticNet(max_iter=5000)
    if "KNN Regressor" in selected_models:
        models["KNN"] = KNeighborsRegressor()
    if "Decision Tree Regressor" in selected_models:
        models["Decision Tree"] = DecisionTreeRegressor(random_state=random_state)
    if "Random Forest Regressor" in selected_models:
        models["Random Forest"] = RandomForestRegressor(n_estimators=300, random_state=random_state)
    if "Gradient Boosting Regressor" in selected_models:
        models["Gradient Boosting"] = GradientBoostingRegressor(random_state=random_state)
    if "SVM Regressor" in selected_models:
        models["SVM"] = SVR()
    if "Neural Network Regressor (MLP) — Advanced" in selected_models:
        models["Neural Network (MLP)"] = MLPRegressor(
            hidden_layer_sizes=(hidden_units,),
            activation=activation,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    cv = _make_cv(cv_choice, n_splits, task="regression") if enable_tuning else None
    groups = model_df.loc[X_train.index, group_col] if enable_tuning and cv_choice == "GroupKFold" and group_col else None

    results = []
    fitted = {}
    predictions_store = pd.DataFrame(index=X_test.index)
    predictions_store["Actual"] = y_test

    progress = st.progress(0)
    for i, (name, model) in enumerate(models.items(), start=1):
        try:
            steps = [("preprocess", preprocessor)]
            if _needs_dense(name):
                steps.append(("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)))
            steps.append(("model", model))
            pipe = Pipeline(steps=steps)

            best_params = "Not tuned"
            if enable_tuning:
                param_grid = _regression_param_grid(name)
                if param_grid:
                    search = GridSearchCV(
                        pipe,
                        param_grid=param_grid,
                        cv=cv,
                        scoring="neg_root_mean_squared_error",
                        n_jobs=-1,
                    )
                    search.fit(X_train, y_train, groups=groups)
                    pipe = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipe.fit(X_train, y_train)
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            train_pred = pipe.predict(X_train)
            row = {
                "Model": name,
                "Train R²": r2_score(y_train, train_pred),
                "Test R²": r2_score(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": _safe_rmse(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "Best Params": str(best_params),
            }
            results.append(row)
            fitted[name] = pipe
            predictions_store[f"{name} predicted"] = y_pred
        except Exception as e:
            st.warning(f"{name} could not run: {e}")
        progress.progress(i / max(len(models), 1))
    progress.empty()

    if not results:
        st.error("No model could be fitted.")
        return

    results_df = pd.DataFrame(results)
    display_df = results_df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    st.markdown("### Model comparison")
    st.dataframe(display_df, use_container_width=True)

    best_idx = results_df["RMSE"].astype(float).idxmin()
    best_row = results_df.loc[best_idx]
    best_model_name = best_row["Model"]
    st.success(_simple_note_for_regression(best_row))

    best_pipe = fitted[best_model_name]
    best_pred = best_pipe.predict(X_test)
    plot_df = pd.DataFrame({"Actual": y_test, "Predicted": best_pred})
    fig = px.scatter(plot_df, x="Actual", y="Predicted", title=f"Actual vs Predicted: {best_model_name}")
    min_v = float(np.nanmin([plot_df["Actual"].min(), plot_df["Predicted"].min()]))
    max_v = float(np.nanmax([plot_df["Actual"].max(), plot_df["Predicted"].max()]))
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash"))
    fig.update_layout(template=plot_template)
    st.plotly_chart(fig, use_container_width=True)

    residual_df = pd.DataFrame({"Predicted": best_pred, "Residual": y_test - best_pred})
    fig_res = px.scatter(residual_df, x="Predicted", y="Residual", title=f"Prediction Residuals: {best_model_name}")
    fig_res.add_hline(y=0, line_dash="dash")
    fig_res.update_layout(template=plot_template)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("### Feature importance / coefficient importance")
    try:
        feature_names = _get_feature_names(
            best_pipe.named_steps["preprocess"],
            numeric_features,
            categorical_features,
            text_features,
        )
        imp = _extract_importance(best_pipe, feature_names, best_model_name)
        if imp is not None and not imp.empty:
            fig_imp = px.bar(
                imp.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top features: {best_model_name}",
            )
            fig_imp.update_layout(template=plot_template, height=650)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.dataframe(imp.round(5), use_container_width=True)
        else:
            st.info("This model does not provide direct feature importance. Use Random Forest/Gradient Boosting or add permutation importance later.")
    except Exception:
        st.info("Feature importance could not be calculated for this model.")

    try:
        _render_advanced_explainability(best_pipe, X_test, y_test, "regression", feature_names, plot_template)
    except Exception:
        pass

    csv = predictions_store.reset_index(drop=False).to_csv(index=False).encode("utf-8")
    st.download_button("Download test-set predictions", csv, file_name="ml_regression_predictions.csv", mime="text/csv")



# ============================================================
# Optional explainability helpers
# ============================================================
def _render_advanced_explainability(best_pipe, X_test, y_test, task, feature_names, plot_template):
    """Model explanation methods matched to model family."""
    model_name = best_pipe.named_steps["model"].__class__.__name__
    display_name = model_name
    # Convert sklearn class names into the simpler names used in the UI.
    name_map = {
        "LogisticRegression": "Logistic Regression",
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "Elastic Net",
        "DecisionTreeClassifier": "Decision Tree",
        "DecisionTreeRegressor": "Decision Tree",
        "RandomForestClassifier": "Random Forest",
        "RandomForestRegressor": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
        "GradientBoostingRegressor": "Gradient Boosting",
        "KNeighborsClassifier": "KNN",
        "KNeighborsRegressor": "KNN",
        "SVC": "SVM",
        "SVR": "SVM",
        "GaussianNB": "Naive Bayes",
        "MLPClassifier": "Neural Network (MLP)",
        "MLPRegressor": "Neural Network (MLP)",
    }
    display_name = name_map.get(model_name, model_name)

    with st.expander("Feature impact analysis", expanded=False):
        st.caption(
            "These methods explain which variables influence prediction. They are not performance metrics. "
            "Use them after checking test-set performance."
        )
        st.info(_explainability_recommendation(display_name))

        st.markdown(
            """
            **Which method should I trust?**
            - **Tree / coefficient importance:** fast and useful, but model-specific.
            - **Permutation importance:** works with almost any model and shows test-set impact by shuffling each original variable.
            - **SHAP:** strongest for detailed explanation in tree-based and linear models, but can be slower.
            """
        )

        if st.checkbox("Run permutation importance", value=False, key=f"ml_perm_{task}"):
            try:
                if task == "classification":
                    scoring = "roc_auc" if len(pd.Series(y_test).unique()) == 2 else "f1_weighted"
                else:
                    scoring = "neg_root_mean_squared_error"
                pi = permutation_importance(
                    best_pipe,
                    X_test,
                    y_test,
                    scoring=scoring,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1,
                )
                p_imp = pd.DataFrame({
                    "Feature": X_test.columns,
                    "Permutation importance": pi.importances_mean,
                    "Std": pi.importances_std,
                }).sort_values("Permutation importance", ascending=False).head(25)
                fig_pi = px.bar(
                    p_imp.sort_values("Permutation importance"),
                    x="Permutation importance",
                    y="Feature",
                    orientation="h",
                    title="Permutation importance — original variables on test set",
                )
                fig_pi.update_layout(template=plot_template, height=650)
                st.plotly_chart(fig_pi, use_container_width=True)
                st.dataframe(p_imp.round(5), use_container_width=True)
                st.caption(
                    "Permutation importance = how much model performance drops when this variable is shuffled. "
                    "Larger positive values mean the variable is more important for prediction."
                )
            except Exception as e:
                st.warning("Permutation importance could not run for this model/data.")
                st.code(str(e))

        if _supports_shap_fast(display_name):
            if st.checkbox("Run SHAP global importance", value=False, key=f"ml_shap_{task}"):
                try:
                    import shap
                    X_trans = best_pipe.named_steps["preprocess"].transform(X_test)
                    X_trans = _to_dense(X_trans)
                    model = best_pipe.named_steps["model"]
                    sample_n = min(200, X_trans.shape[0])
                    X_sample = X_trans[:sample_n]
                    explainer = shap.Explainer(model, X_sample)
                    shap_values = explainer(X_sample)

                    vals = shap_values.values
                    if vals.ndim == 3:
                        vals = np.mean(np.abs(vals), axis=(0, 2))
                    else:
                        vals = np.mean(np.abs(vals), axis=0)
                    n = min(len(feature_names), len(vals))
                    shap_df = pd.DataFrame({"Feature": feature_names[:n], "Mean |SHAP|": vals[:n]})
                    shap_df = shap_df.sort_values("Mean |SHAP|", ascending=False).head(25)
                    fig_shap = px.bar(
                        shap_df.sort_values("Mean |SHAP|"),
                        x="Mean |SHAP|",
                        y="Feature",
                        orientation="h",
                        title="SHAP global importance — transformed features",
                    )
                    fig_shap.update_layout(template=plot_template, height=650)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.dataframe(shap_df.round(5), use_container_width=True)
                    st.caption(
                        "Mean |SHAP| ranks features by average contribution size. "
                        "It explains prediction impact, not whether the model is accurate."
                    )
                except ImportError:
                    st.info("SHAP is listed in requirements.txt, but it is not installed in this environment yet. Run: pip install -r requirements.txt")
                except Exception as e:
                    st.warning("SHAP could not run for this model/data. Use permutation importance instead.")
                    st.code(str(e))
        else:
            st.info(
                f"SHAP is not enabled for {display_name} in this app version. "
                "For KNN, SVM, Naive Bayes, and Neural Networks, use permutation importance because it is more general and safer here."
            )

# ============================================================
# Clustering and PCA
# ============================================================
def _cluster_profile_table(clustered, data, predictor_cols, label_col="Cluster"):
    num_cols_original = data.loc[clustered.index, predictor_cols].select_dtypes(include=[np.number]).columns.tolist()
    if num_cols_original:
        return clustered.groupby(label_col)[num_cols_original].mean().round(3)
    return pd.DataFrame()


def _render_cluster_outputs(clustered, labels, X_processed, predictor_cols, data, plot_template, title_prefix, noise_label=None):
    """Shared outputs for clustering models."""
    unique_labels = pd.Series(labels).astype(str).nunique()
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows used", f"{X_processed.shape[0]:,}")
    m2.metric("Number of clusters", unique_labels)

    sil = np.nan
    try:
        # Silhouette needs at least two labels and less labels than observations.
        if unique_labels > 1 and unique_labels < X_processed.shape[0]:
            sil = silhouette_score(X_processed, labels)
    except Exception:
        pass
    m3.metric("Silhouette score", "NA" if pd.isna(sil) else f"{sil:.3f}")

    if pd.notna(sil):
        if sil >= 0.50:
            st.success("Silhouette is relatively high, suggesting better-separated clusters.")
        elif sil >= 0.25:
            st.info("Silhouette is moderate. Check PCA plot and cluster profiles before accepting the clusters.")
        else:
            st.warning("Silhouette is low. Clusters may overlap; try different features, scaling, or another algorithm.")
    st.caption("Silhouette ranges from -1 to 1. Higher is better; negative values suggest poor assignment.")

    st.markdown("### Cluster sizes")
    size_df = clustered["Cluster"].value_counts().sort_index().reset_index()
    size_df.columns = ["Cluster", "Count"]
    st.dataframe(size_df, use_container_width=True)
    fig_size = px.bar(size_df, x="Cluster", y="Count", title=f"{title_prefix} cluster sizes")
    fig_size.update_layout(template=plot_template)
    st.plotly_chart(fig_size, use_container_width=True)

    if noise_label is not None and str(noise_label) in set(clustered["Cluster"].astype(str)):
        noise_n = int((clustered["Cluster"].astype(str) == str(noise_label)).sum())
        st.info(f"DBSCAN marked {noise_n:,} observation(s) as noise/outliers using label {noise_label}.")

    st.markdown("### PCA 2D visualization")
    try:
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X_processed)
        pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "Cluster": clustered["Cluster"].astype(str).values})
        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=f"{title_prefix} clusters on PCA plot (PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%})",
        )
        fig_pca.update_layout(template=plot_template)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("PCA is used only for 2D visualization. The clustering model uses the selected preprocessed features.")
    except Exception:
        st.info("PCA plot could not be created.")

    st.markdown("### Cluster profile")
    profile_num = _cluster_profile_table(clustered, data, predictor_cols)
    if not profile_num.empty:
        st.dataframe(profile_num, use_container_width=True)
        st.caption("Cluster profile shows the average value of original numeric features within each cluster. Use it to name and interpret clusters.")
    else:
        st.info("No numeric columns available for mean cluster profile.")

    csv = clustered.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset with clusters", csv, file_name="ml_clustered_dataset.csv", mime="text/csv")


def _render_clustering(data, predictor_cols, plot_template):
    st.markdown("### Clustering / PCA — unsupervised learning")
    st.caption("Use this when there is no target/label. The goal is to group similar observations or explore hidden structure.")

    with st.expander("Clustering setup", expanded=True):
        st.markdown(
            """
            Clustering is exploratory: there is no true target column. The app groups observations based on selected features, then helps you judge whether the groups are useful.

            - **K-Means:** fast and useful when clusters are fairly compact; requires choosing K.
            - **DBSCAN:** density-based; does not require K and can detect noise/outliers.
            - **Hierarchical clustering:** useful for small/medium datasets and relationship exploration using a dendrogram.

            Key checks: **Inertia/SSE** for K-Means, **Elbow plot**, **Silhouette score**, **PCA visualization**, and **cluster profile**.
            """
        )

    if not predictor_cols:
        st.warning("Select predictors/features first.")
        return

    X = data[predictor_cols].copy()
    X = X.dropna(how="all")
    clustering_audit = build_model_data_audit(
        data,
        X,
        predictor_cols,
        "Rows missing every selected predictor are excluded; remaining predictor missingness is imputed inside the Pipeline.",
    )
    render_model_data_audit(clustering_audit)

    categorical_encoding, scale_numeric, text_features, tfidf_max_features = _preprocessing_options(
        data=data.loc[X.index],
        predictor_cols=predictor_cols,
        prefix="ml_clu",
    )
    if not scale_numeric:
        st.warning("For distance-based clustering, StandardScaler is usually recommended because scale changes distances.")

    preprocessor, numeric_features, categorical_features, text_features = _build_preprocessor(
        X,
        scale_numeric=scale_numeric,
        categorical_encoding=categorical_encoding,
        text_features=text_features,
        tfidf_max_features=tfidf_max_features,
    )

    try:
        X_processed = preprocessor.fit_transform(X)
        X_processed = _to_dense(X_processed)
    except Exception as e:
        st.error("Could not preprocess the selected features for clustering.")
        st.code(str(e))
        return

    if X_processed.shape[0] < 5:
        st.warning("Need at least 5 rows for useful clustering.")
        return

    algorithm = st.radio(
        "Clustering algorithm",
        ["K-Means", "DBSCAN", "Hierarchical Clustering"],
        horizontal=True,
        key="ml_cluster_algo",
    )

    random_state = 42

    if algorithm == "K-Means":
        st.info("K-Means needs K in advance. It works best when clusters are compact and roughly spherical, and it can be sensitive to outliers.")
        c1, c2 = st.columns(2)
        k = c1.slider("Number of clusters (K)", 2, min(12, max(2, X_processed.shape[0] - 1)), 3, key="ml_kmeans_k")
        random_state = int(c2.number_input("Random seed", value=42, step=1, key="ml_cluster_seed"))

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_processed)

        clustered = data.loc[X.index].copy()
        clustered["Cluster"] = labels.astype(str)

        e1, e2 = st.columns(2)
        e1.metric("Inertia / SSE", f"{kmeans.inertia_:.3f}")
        e2.caption("Inertia/SSE is the sum of squared distances to cluster centroids. Lower is better, but it naturally decreases as K increases, so use the elbow plot.")

        with st.expander("Elbow plot — choose K", expanded=True):
            max_k = min(12, X_processed.shape[0] - 1)
            if max_k >= 2:
                inertias = []
                silhouettes = []
                ks = list(range(2, max_k + 1))
                for kk in ks:
                    km = KMeans(n_clusters=kk, random_state=random_state, n_init=10)
                    kk_labels = km.fit_predict(X_processed)
                    inertias.append(km.inertia_)
                    try:
                        silhouettes.append(silhouette_score(X_processed, kk_labels))
                    except Exception:
                        silhouettes.append(np.nan)
                elbow_df = pd.DataFrame({"K": ks, "Inertia / SSE": inertias, "Silhouette": silhouettes})
                fig_elbow = px.line(elbow_df, x="K", y="Inertia / SSE", markers=True, title="Elbow plot: K vs Inertia/SSE")
                fig_elbow.update_layout(template=plot_template)
                st.plotly_chart(fig_elbow, use_container_width=True)
                st.dataframe(elbow_df.round(4), use_container_width=True)
                st.caption("Choose the elbow point where SSE starts decreasing slowly. Also check Silhouette; higher is better.")

        _render_cluster_outputs(clustered, labels, X_processed, predictor_cols, data, plot_template, "K-Means")

    elif algorithm == "DBSCAN":
        st.info("DBSCAN is density-based. It does not need K and can automatically label sparse points as noise/outliers.")
        d1, d2 = st.columns(2)
        eps = d1.slider("eps — neighborhood radius", 0.05, 10.0, 0.5, 0.05, key="ml_dbscan_eps")
        min_samples = d2.slider("min_samples", 2, 50, 5, key="ml_dbscan_min_samples")

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_processed)
        labels_str = np.where(labels == -1, "Noise / outlier", labels.astype(str))

        clustered = data.loc[X.index].copy()
        clustered["Cluster"] = labels_str

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters <= 1:
            st.warning("DBSCAN found one or zero real clusters. Try increasing eps, lowering min_samples, or scaling/checking your features.")
        _render_cluster_outputs(clustered, labels, X_processed, predictor_cols, data, plot_template, "DBSCAN", noise_label="Noise / outlier")

    elif algorithm == "Hierarchical Clustering":
        st.info("Hierarchical clustering builds a tree of relationships. It is best for small to medium datasets; dendrograms may be slow for large data.")
        h1, h2 = st.columns(2)
        n_clusters = h1.slider("Number of clusters", 2, min(12, max(2, X_processed.shape[0] - 1)), 3, key="ml_hier_k")
        linkage_method = h2.selectbox("Linkage", ["ward", "complete", "average", "single"], key="ml_hier_linkage")

        try:
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            labels = hc.fit_predict(X_processed)
        except Exception as e:
            st.error("Hierarchical clustering could not run with these settings.")
            st.code(str(e))
            return

        clustered = data.loc[X.index].copy()
        clustered["Cluster"] = labels.astype(str)

        with st.expander("Dendrogram", expanded=False):
            if X_processed.shape[0] > 300:
                st.info("Dendrogram is hidden for datasets larger than 300 rows to avoid slow rendering. Sample the data if needed.")
            else:
                try:
                    from scipy.cluster.hierarchy import linkage
                    Z = linkage(X_processed, method=linkage_method)
                    fig_dend = ff.create_dendrogram(X_processed, linkagefun=lambda _: Z)
                    fig_dend.update_layout(title="Hierarchical clustering dendrogram", template=plot_template, height=650)
                    st.plotly_chart(fig_dend, use_container_width=True)
                    st.caption("The dendrogram helps you see how observations/clusters merge. Cut the tree at different heights to choose cluster count.")
                except Exception as e:
                    st.info("Dendrogram could not be created.")
                    st.code(str(e))

        _render_cluster_outputs(clustered, labels, X_processed, predictor_cols, data, plot_template, "Hierarchical")


# ============================================================
# Main renderer
# ============================================================
def render_machine_learning_tab(df, df_cleaned=None, plot_template="plotly_white"):
    st.markdown("## 🤖 Machine Learning")
    st.caption("Prediction-focused workflow: EDA, X/Y, split, preprocessing, Pipeline, GridSearchCV, CV, metrics, threshold tuning, and final model selection.")

    with st.expander("When to use this section", expanded=True):
        st.markdown(
            """
            Use **Machine Learning** when the main goal is **prediction on new data** or **pattern discovery**.

            - **Classification:** target is binary/categorical. Example: readmission yes/no, disease class, mortality yes/no.
            - **Regression:** target is numeric. Example: length of stay, cost, score, blood pressure.
            - **Clustering/PCA:** no target; explore groups or structure.

            Difference from Base Models: Base Models focus on **coefficients, p-values, and interpretation**. Machine Learning focuses on **test-set performance, cross-validation, tuning, and prediction**.

            Neural Network in this version uses **MLP from scikit-learn** to keep installation simple. It follows the same core ANN ideas: input layer, hidden layers, activation functions, fitting, prediction, validation/early stopping, and test-set evaluation.
            """
        )

    data_source = st.radio(
        "Data source",
        ["Original data", "Cleaned data"],
        index=1 if df_cleaned is not None else 0,
        horizontal=True,
        key="ml_data_source",
    )
    data = _get_working_df(df, df_cleaned) if data_source == "Cleaned data" else df

    if data is None or data.empty:
        st.warning("Upload a dataset first.")
        return

    all_cols = data.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Need at least two columns for machine learning.")
        return

    st.markdown("### Step 1 — Select X and Y")
    target_options = ["No target — clustering / PCA"] + all_cols
    target_col = st.selectbox("Y / target column", target_options, key="ml_target_col")

    default_predictors = [c for c in all_cols if c != target_col]
    predictor_cols = st.multiselect(
        "X / predictor feature columns",
        [c for c in all_cols if c != target_col],
        default=default_predictors[: min(8, len(default_predictors))],
        key="ml_predictor_cols",
    )

    inferred = _infer_task_type(data, target_col)
    task_type = st.selectbox(
        "Task type",
        ["Auto-detect", "Classification", "Regression", "Clustering / Unsupervised"],
        key="ml_task_type",
    )
    if task_type == "Auto-detect":
        task_type = inferred

    st.info(f"Detected/selected task: **{task_type}**")

    if task_type == "Not enough target variation":
        st.error("The selected target has fewer than two unique values. Choose a different target.")
        return

    if not predictor_cols:
        st.warning("Select at least one predictor/feature column.")
        return

    st.markdown("---")
    st.caption(
        f"The selected workflow will receive all {len(data):,} source rows. "
        "Eligibility, Train/Test allocation, and exclusions are reported below."
    )
    if not st.button(
        "▶ Run Machine Learning Workflow",
        use_container_width=True,
        type="primary",
        key="ml_run_workflow",
    ):
        st.info("Configure X, Y, and the task, then run the workflow explicitly.")
        return

    try:
        if task_type == "Classification":
            if target_col == "No target — clustering / PCA":
                st.warning("Classification needs a target column.")
                return
            _render_classification(data, target_col, predictor_cols, plot_template)
        elif task_type == "Regression":
            if target_col == "No target — clustering / PCA":
                st.warning("Regression needs a target column.")
                return
            _render_regression(data, target_col, predictor_cols, plot_template)
        elif task_type == "Clustering / Unsupervised":
            _render_clustering(data, predictor_cols, plot_template)
        else:
            st.error("Unknown task type.")
    except Exception as e:
        st.error("Machine Learning module encountered an error.")
        st.code(str(e))
