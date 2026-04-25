import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import importlib
from matplotlib import patches

def mutual_info(df, target):
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' was not found in the dataframe.")

    data = df.dropna()  # Drop rows with missing values

    features = data.drop(target, axis=1).copy()
    categorical_features = features.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for column in categorical_features:
        features[column] = pd.Categorical(features[column]).codes

    X = features
    y = data[target]
    discrete_features = [column in categorical_features for column in X.columns]

    if categorical_features:
        print(
            "Encoded categorical columns for mutual_info: "
            f"{', '.join(categorical_features)}"
        )

    # Compute mutual information between each feature and the target
    mi = mutual_info_classif(X, y, random_state=0, discrete_features=discrete_features)

    # Show ranked features
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print(mi_scores.head(10))

    return mi_scores

def annotated_heatmap(df,  dfname, column_to_start, column_to_end, threshold):
    df=df.select_dtypes(include=['number'])

    df_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_partial = df[df_columns].iloc[:, column_to_start:column_to_end]
    plt.figure(figsize=(20, 12))
    corr = df_partial.corr()
    
    ax = sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True)
    ax.set_title(f"Correlation Heatmap for {dfname} Dataset (Columns {column_to_start} to {column_to_end}) with a threshold set at {threshold:.0%} or above highlighted")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i != j and abs(corr.iloc[i, j]) >= threshold:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="lime", lw=3))

    plt.show()

def heatmap(heatmap_type):
    if heatmap_type == "annotated":
        def annotated_heatmap(df,  dfname, column_to_start, column_to_end, threshold):
            df=df.select_dtypes(include=['number'])

            df_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            df_partial = df[df_columns].iloc[:, column_to_start:column_to_end]
            plt.figure(figsize=(20, 12))
            corr = df_partial.corr()
            
            ax = sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True)
            ax.set_title(f"Correlation Heatmap for {dfname} Dataset (Columns {column_to_start} to {column_to_end}) with a threshold set at {threshold:.0%} or above highlighted")

            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    if i != j and abs(corr.iloc[i, j]) >= threshold:
                        ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="lime", lw=3))

            plt.show()
    elif heatmap_type == "selected columns":
        def selected_col_heatmap(df, dfname, cols_2_include):
            # change dataframe to only include numeric columns for correlation heatmap
            df = df[cols_2_include].select_dtypes(include=['number'])
            df_partial = df[cols_2_include]
            # change figure size
            fig, ax = plt.subplots(figsize=(12, 10))
            # create the heatmap
            ax = sns.heatmap(df_partial.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
            # create the title
            ax.set_title(f'Correlation Heatmap for {dfname} Dataset (Selected Columns)', fontsize=16)
            # save the figure
            plt.savefig(f'{dfname}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
    else:
        raise ValueError(f"Invalid heatmap type: {heatmap_type}. Choose 'annotated' or 'selected columns'.")
    
def selected_col_heatmap(df, dfname, cols_2_include):
    # change dataframe to only include numeric columns for correlation heatmap
    df = df[cols_2_include].select_dtypes(include=['number'])
    df_partial = df[cols_2_include]
    # change figure size
    fig, ax = plt.subplots(figsize=(12, 10))
    # create the heatmap
    ax = sns.heatmap(df_partial.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    # create the title
    ax.set_title(f'Correlation Heatmap for {dfname} Dataset (Selected Columns)', fontsize=16)
    # save the figure
    plt.savefig(f'{dfname}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def Feature_selection(df, features, target):
    # Final Demo Version - Swarm Plot

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # --- 1. TRAIN MODEL & PREPARE DATA ---

    df = df.dropna()  # Drop rows with missing values for simplicity

    # Get importance
    X = df[features]
    y = df[target]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create ranking
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # --- 2. VISUALIZE IMPORTANCE THROUGH DATA DISTRIBUTION ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot each feature's distribution by species
    features = importance_df['feature'].tolist()
    colors = sns.color_palette("crest", n_colors=len(features))

    for idx, (ax, feature, color, rank) in enumerate(zip(axes.flatten(), features, colors, range(1, 5))):
        # Create swarm plot
        sns.swarmplot(data=df, x=target[0], y=feature, 
                    hue=target[0], palette='Set2', size=3, ax=ax)
        
        # Add importance annotation
        imp_value = importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0]
        ax.text(0.02, 0.98, f'Rank #{rank}\n({imp_value:.1%})', 
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Measurement (mm/g)')
        ax.legend().remove()

    plt.suptitle('Feature Importance Visualization: Actual Data Distributions\n' +
                'Higher-ranked features show clearer separation', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # --- 3. CONSOLE SUMMARY ---
    print("="*60)
    print("FEATURE IMPORTANCE RANKING FROM MODEL")
    print("="*60)
    for i, (feature, imp) in enumerate(zip(importance_df['feature'], importance_df['importance']), 1):
        separation = "HIGH" if imp > 0.2 else "MODERATE" if imp > 0.15 else "LOW"
        print(f"{i}. {feature:<20} {imp:.1%}  →  {target[0]} separation: {separation}")
        
def pca_evaluate_dataset(df, dataset_name, target_col=None, variance_threshold=0.80):
    """
   I used the following code to invoke this function and return both a summary and plots
   (my datasets are below, but you can replace with your own. Note that each entry must be comprised 
   of a dataframe, a name, and a target column):
   
    datasets_to_evaluate = [
    ("students", students, "target"),
    ("dcd25_renamed", dcd25_renamed, "testscore_gap"),
    ("sid", sid, "sal_parity55"),
    ("joined", joined, None),
]

summaries = []
plot_results = {}

for dataset_name, dataset_df, target_col in datasets_to_evaluate:
    result = pca_evaluate_dataset(dataset_df, dataset_name, target_col=target_col)
    if result is None:
        continue

    summary, plot_df = result
    summaries.append(summary)
    plot_results[dataset_name] = {"plot_df": plot_df, "bubble_field": summary["bubble_field"]}

    print(f"{dataset_name}: PC1={summary['pc1_variance_pct']:.2f}%, PC2={summary['pc2_variance_pct']:.2f}%")
    print(f"  PC1+PC2={summary['pc1_pc2_cumulative_pct']:.2f}%, components for 80% variance={summary['pcs_for_80pct_variance']}")

if summaries:
    pca_summary_df = pd.DataFrame(summaries).sort_values("dataset")
    print("\nPCA evaluation summary:")
    print(pca_summary_df.to_string(index=False))
else:
    print("No numeric datasets were available for PCA evaluation.")

if plot_results:
    n = len(plot_results)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, (dataset_name, result) in zip(axes, plot_results.items()):
        plot_df = result["plot_df"]
        bubble_field = result["bubble_field"]

        sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            size=bubble_field,
            sizes=(20, 400),
            alpha=0.6,
            color="teal",
            legend=False,
            ax=ax,
        )
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{dataset_name}: PC1 vs PC2 bubble plot")

    for ax in axes[n:]:
        ax.axis("off")

    plt.show()


    """
    from sklearn.preprocessing import StandardScaler
    from sklearn import decomposition
    
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Keep target only for bubble sizing; exclude from PCA features to avoid leakage.
    bubble_label = target_col if target_col in df.columns else "bubble_size"

    if target_col is not None and target_col in numeric_df.columns:
        feature_df = numeric_df.drop(columns=[target_col])
    else:
        feature_df = numeric_df

    if target_col is not None and target_col in df.columns:
        aligned = pd.concat([feature_df, df[target_col]], axis=1).dropna()
        X = aligned[feature_df.columns]
        bubble_series = aligned[target_col]
    else:
        X = feature_df.dropna()
        bubble_series = pd.Series(np.ones(len(X)), index=X.index, name="bubble_size")

    if X.shape[0] == 0 or X.shape[1] < 2:
        print(f"{dataset_name}: not enough numeric predictor columns for PCA")
        return None

    scaled_values = StandardScaler().fit_transform(X)

    pca = decomposition.PCA()
    pca.fit(scaled_values)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pcs_needed = int(np.argmax(cumulative >= variance_threshold) + 1) if np.any(cumulative >= variance_threshold) else len(explained)

    pca_scores = pca.transform(scaled_values)
    plot_df = pd.DataFrame(
        {
            "PC1": pca_scores[:, 0],
            "PC2": pca_scores[:, 1],
            bubble_label: bubble_series.to_numpy(),
        }
    )

    summary = {
        "dataset": dataset_name,
        "rows": X.shape[0],
        "numeric_features": X.shape[1],
        "pc1_variance_pct": explained[0] * 100,
        "pc2_variance_pct": explained[1] * 100 if len(explained) > 1 else np.nan,
        "pc1_pc2_cumulative_pct": cumulative[1] * 100 if len(cumulative) > 1 else explained[0] * 100,
        "pcs_for_80pct_variance": pcs_needed,
        "bubble_field": bubble_label,
    }

    return summary, plot_df

def RF_regressor(df, dfname, feature_cols, target_col):
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error
    import warnings
    
    # Suppress pandas warning about numexpr minimum version
    warnings.filterwarnings("ignore", message="Pandas requires version '2.10.2' or newer of 'numexpr'")

    reg_df = df[feature_cols + [target_col]].dropna()
    X = reg_df[feature_cols]
    y = reg_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_reg_df = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42
    )
    rf_reg_df.fit(X_train, y_train)

    y_pred = rf_reg_df.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"RandomForestRegressor RMSE on {dfname}: {rmse:.4f}")
    
def kfold_cross_val(df,dfname,feature_cols,target_col):
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import KFold

    # Suppress pandas warning about numexpr minimum version
    warnings.filterwarnings("ignore", message="Pandas requires version '2.10.2' or newer of 'numexpr'")

    df.attrs["name"] = dfname

    reg_df = df[feature_cols + [target_col]].dropna()
    X = reg_df[feature_cols]
    y = reg_df[target_col]

    # Define model settings and 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    total_folds = cv.get_n_splits()

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        rmse_scores.append(root_mean_squared_error(y.iloc[test_idx], y_pred))

    rmse_scores = np.array(rmse_scores)

    print("5-Fold RMSE scores:")
    for i, score in enumerate(rmse_scores, start=1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean RMSE: {rmse_scores.mean():.4f}")
    print(f"Std RMSE: {rmse_scores.std():.4f}")

    # Plot fold-wise RMSE and overall average
    plt.figure(figsize=(8, 4.5))
    plt.bar(range(1, 6), rmse_scores, color="#2a9d8f", edgecolor="black")
    plt.axhline(rmse_scores.mean(), color="#e76f51", linestyle="--", linewidth=2, label=f"Mean RMSE = {rmse_scores.mean():.4f}")
    plt.xticks(range(1, 6), [f"Fold {i}" for i in range(1, 6)])
    plt.ylabel("RMSE")
    plt.title(f"5-Fold Cross-Validation RMSE for {df.attrs.get('name', 'Dataset')}")
    plt.legend()
    plt.tight_layout()
    plt.show();

def tree_mode(df, feature_cols, target_col, mode="regressor"):

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
        root_mean_squared_error,
    )

    mode_normalized = mode.strip().lower()

    if mode_normalized in ("classifier", "classification"):
        X_cmp = df[feature_cols].dropna()
        y_cmp = (df.loc[X_cmp.index, target_col] > 0).astype(int)

        stratify_y = y_cmp if y_cmp.nunique() > 1 else None
        X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
            X_cmp, y_cmp, test_size=0.2, random_state=42, stratify=stratify_y
        )

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train_cmp, y_train_cmp)
        y_pred = model.predict(X_test_cmp)

        if y_test_cmp.nunique() > 1:
            y_proba = model.predict_proba(X_test_cmp)[:, 1]
            roc_auc = float(roc_auc_score(y_test_cmp, y_proba))
        else:
            roc_auc = float("nan")

        results = {
            "mode": "classification",
            "accuracy": float(accuracy_score(y_test_cmp, y_pred)),
            "f1": float(f1_score(y_test_cmp, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "n_train": int(len(X_train_cmp)),
            "n_test": int(len(X_test_cmp)),
        }

        print(f"RandomForestClassifier Accuracy: {results['accuracy']:.4f}")
        print(f"RandomForestClassifier F1: {results['f1']:.4f}")
        print(f"RandomForestClassifier ROC-AUC: {results['roc_auc']:.4f}")
        return results

    if mode_normalized in ("regressor", "regression"):
        reg_df = df[feature_cols + [target_col]].dropna()
        X_cmp = reg_df[feature_cols]
        y_cmp = reg_df[target_col]

        X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
            X_cmp, y_cmp, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train_cmp, y_train_cmp)
        y_pred = model.predict(X_test_cmp)

        results = {
            "mode": "regression",
            "rmse": float(root_mean_squared_error(y_test_cmp, y_pred)),
            "mae": float(mean_absolute_error(y_test_cmp, y_pred)),
            "r2": float(r2_score(y_test_cmp, y_pred)),
            "n_train": int(len(X_train_cmp)),
            "n_test": int(len(X_test_cmp)),
        }

        print(f"RandomForestRegressor RMSE: {results['rmse']:.4f}")
        print(f"RandomForestRegressor MAE: {results['mae']:.4f}")
        print(f"RandomForestRegressor R2: {results['r2']:.4f}")
        return results

    raise ValueError("Invalid mode. Choose 'classifier'/'classification' or 'regressor'/'regression'.")
        
def compare_rf_models(df, feature_cols, target_col):

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error

    X_cmp = df[feature_cols].dropna()
    y_cmp = (df.loc[X_cmp.index, target_col] > 0).astype(int)

    X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
        X_cmp, y_cmp, test_size=0.2, random_state=42, stratify=y_cmp
    )

    rf_clf_cmp = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_clf_cmp.fit(X_train_cmp, y_train_cmp)
    y_pred_clf_cmp = rf_clf_cmp.predict(X_test_cmp)
    rmse_clf = root_mean_squared_error(y_test_cmp, y_pred_clf_cmp)

    rf_reg_cmp = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_reg_cmp.fit(X_train_cmp, y_train_cmp)
    y_pred_reg_cmp = rf_reg_cmp.predict(X_test_cmp)
    rmse_reg = root_mean_squared_error(y_test_cmp, y_pred_reg_cmp)

    print(f"RandomForestClassifier RMSE: {rmse_clf:.4f}")
    print(f"RandomForestRegressor RMSE: {rmse_reg:.4f}")
    print("Lower RMSE:", "Classifier" if rmse_clf < rmse_reg else "Regressor")


def compare_rf_models_sid(
    df,
    feature_cols,
    target_col,
    *,
    classification_threshold=None,
    test_size=0.2,
    random_state=42,
    n_estimators=200,
    max_depth=10,
):
    """Compare RF classifier and regressor on SID with separate targets.

    The classifier is trained on a binary label derived from the target using
    ``classification_threshold``. If no threshold is provided, the median of the
    target is used. The regressor is trained on the raw continuous target.
    """

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split

    model_df = df[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y_reg = model_df[target_col]

    if classification_threshold is None:
        classification_threshold = float(y_reg.median())

    y_cls = (y_reg > classification_threshold).astype(int)
    if y_cls.nunique() < 2:
        raise ValueError(
            "Classification target collapsed to one class after thresholding. "
            "Choose a different classification_threshold."
        )

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=test_size,
        random_state=random_state,
        stratify=y_cls,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf_clf.fit(X_train, y_cls_train)
    y_pred_cls = rf_clf.predict(X_test)

    rf_reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf_reg.fit(X_train, y_reg_train)
    y_pred_reg = rf_reg.predict(X_test)

    clf_accuracy = accuracy_score(y_cls_test, y_pred_cls)
    clf_f1 = f1_score(y_cls_test, y_pred_cls, zero_division=0)
    reg_rmse = root_mean_squared_error(y_reg_test, y_pred_reg)

    print(f"Classification threshold: {classification_threshold:.4f}")
    print(f"RandomForestClassifier Accuracy: {clf_accuracy:.4f}")
    print(f"RandomForestClassifier F1: {clf_f1:.4f}")
    print(f"RandomForestRegressor RMSE: {reg_rmse:.4f}")

    return {
        "classification_threshold": float(classification_threshold),
        "classifier_accuracy": float(clf_accuracy),
        "classifier_f1": float(clf_f1),
        "regressor_rmse": float(reg_rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

def run_rf_5fold(
    data,
    feature_cols,
    target_col,
    *,
    delimiter=None,
    n_splits=5,
    random_state=42,
    n_estimators=200,
    max_depth=10,
    show_progress=True,
    show_plot=True,
):
    """Run RandomForestRegressor K-fold CV with RMSE reporting and optional plot."""

    # imports
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import KFold
    # from tqdm.auto import tqdm

    warnings.filterwarnings(
        "ignore",
        message="Pandas requires version '2.10.2' or newer of 'numexpr'",
    )


    if isinstance(data, (str, Path)):
        read_csv_kwargs = {"sep": delimiter} if delimiter is not None else {}
        df = pd.read_csv(data, **read_csv_kwargs)
        dataset_name = Path(data).stem
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        dataset_name = df.attrs.get("name", "dataset")
    else:
        raise TypeError("data must be a pandas DataFrame or a file path")

    required_cols = list(feature_cols) + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    reg_df = df[required_cols].dropna()
    X = reg_df[feature_cols]
    y = reg_df[target_col]

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores = []
    total_folds = cv.get_n_splits()

    fold_iter = cv.split(X)
    # if show_progress:
    #     fold_iter = tqdm(fold_iter, total=total_folds, desc="CV Progress", mininterval=1)

    for train_idx, test_idx in fold_iter:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        rmse_scores.append(root_mean_squared_error(y.iloc[test_idx], y_pred))

    rmse_scores = np.array(rmse_scores)

    print("5-Fold RMSE scores:" if n_splits == 5 else f"{n_splits}-Fold RMSE scores:")
    for i, score in enumerate(rmse_scores, start=1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean RMSE: {rmse_scores.mean():.4f}")
    print(f"Std RMSE: {rmse_scores.std():.4f}")

    if show_plot:
        plt.figure(figsize=(8, 4.5))
        plt.bar(range(1, total_folds + 1), rmse_scores, color="#2a9d8f", edgecolor="black")
        plt.axhline(
            rmse_scores.mean(),
            color="#e76f51",
            linestyle="--",
            linewidth=2,
            label=f"Mean RMSE = {rmse_scores.mean():.4f}",
        )
        plt.xticks(range(1, total_folds + 1), [f"Fold {i}" for i in range(1, total_folds + 1)])
        plt.ylabel("RMSE")
        plt.title(f"{n_splits}-Fold Cross-Validation RMSE ({dataset_name})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "rmse_scores": rmse_scores,
        "mean_rmse": float(rmse_scores.mean()),
        "std_rmse": float(rmse_scores.std()),
    }

def show_missing_columns(df, lower_bound, upper_bound):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    filtered_missing = missing_percent[(missing_percent > lower_bound) & (missing_percent <= upper_bound)]
    count = len(filtered_missing)
    
    # Generate markdown table
    table_df = filtered_missing.reset_index()
    table_df.columns = ['Column', 'Missing %']
    table_df['Missing %'] = table_df['Missing %'].round(2)
    print(table_df.to_markdown(index=False))
    print(f"There are \033[1m{count}\033[0m columns with missing values between {lower_bound}% and {upper_bound}% in this dataset.")
    
    return filtered_missing, count

def find_missing(df):
    missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percent': (df.isnull().sum() / len(df) * 100).values
    })
    missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)
    print(missing_summary)
    
    return missing_summary

def find_outliers(dataframe):
    df = dataframe.select_dtypes(include=[np.number])

    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"Outliers in column '{column}':")
        print(outliers[[column]])

def calculate_r2_for_datasets(datasets, target_map, test_size=0.2, random_state=42):
    """Calculate test-set R2 for each dataset in a dict.

    Args:
        datasets: dict[str, pd.DataFrame]
        target_map: dict[str, str] mapping dataset name to target column
        test_size: fraction of rows for the test split
        random_state: split seed for reproducibility

    Returns:
        pd.DataFrame with columns: dataset, r2, note
    """
    ColumnTransformer = importlib.import_module("sklearn.compose").ColumnTransformer
    SimpleImputer = importlib.import_module("sklearn.impute").SimpleImputer
    LinearRegression = importlib.import_module("sklearn.linear_model").LinearRegression
    r2_score = importlib.import_module("sklearn.metrics").r2_score
    train_test_split = importlib.import_module("sklearn.model_selection").train_test_split
    Pipeline = importlib.import_module("sklearn.pipeline").Pipeline
    OneHotEncoder = importlib.import_module("sklearn.preprocessing").OneHotEncoder

    results = []

    for name, df in datasets.items():
        target_col = target_map.get(name)

        if target_col is None:
            results.append({"dataset": name, "r2": None, "note": "No target in target_map"})
            continue

        if target_col not in df.columns:
            results.append({"dataset": name, "r2": None, "note": f"Target '{target_col}' not found"})
            continue

        data = df.copy().dropna(subset=[target_col])
        X = data.drop(columns=[target_col])
        y = data[target_col]

        if len(data) < 3:
            results.append({"dataset": name, "r2": None, "note": "Not enough rows"})
            continue

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        results.append({"dataset": name, "r2": float(r2), "note": "ok"})

    return pd.DataFrame(results).sort_values("r2", ascending=False, na_position="last")


def regplotter(df, feature1, feature1_title, feature2, feature2_title, feature3, feature3_title):
    # Accept either a string column name or a one-item list like ['col_name'].
    f1 = feature1[0] if isinstance(feature1, (list, tuple)) else feature1
    f2 = feature2[0] if isinstance(feature2, (list, tuple)) else feature2
    f3 = feature3[0] if isinstance(feature3, (list, tuple)) else feature3

    featurelist = [f1, f2, f3]
    df_clean = df.dropna(subset=featurelist)

    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 8)

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = sns.scatterplot(
        data=df_clean, 
        x=f1, 
        y=f2,
        hue=feature3,
        palette='viridis', 
        alpha=0.7, 
        s=60,
        edgecolor='k',
        legend=False
    )

    # Add regression line (using all data points, not colored by state)
    reg_line = sns.regplot(
        data=df_clean, 
        x=f1, 
        y=f2, 
        scatter=False,  # Don't show the scatter points again
        color='red', 
        line_kws={'linewidth': 2.5, 'label': 'Regression Line'},
        ci=95,  # Show 95% confidence interval
    )

    # Calculate and display regression statistics.
    x_values = df_clean[f1].to_numpy(dtype=float)
    y_values = df_clean[f2].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    r_value = np.corrcoef(x_values, y_values)[0, 1]
    r_squared = float(r_value ** 2)
    p_value = float("nan")

    # Add text annotation with regression statistics
    text_str = f'Regression Statistics:\nSlope: {slope:.2f}\nR²: {r_squared:.3f}\nP-value: {p_value:.4f}'
    plt.text(0.80, 0.15, text_str, transform=plt.gca().transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title(f'Relationship Between {feature1_title} and {feature2_title} (with Regression Analysis)', fontsize=16)
    plt.xlabel(f'{feature1_title} ({f1})', fontsize=12)
    plt.ylabel(f'{feature2_title} ({f2})', fontsize=12)
    plt.axhline(0, color='darkgray', linestyle='--', linewidth=1.5, label='Break-even Point')

    plt.tight_layout()
    plt.show()

    # Optional: Print detailed regression output
    print("=" * 60)
    print("REGRESSION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Dependent Variable:  {feature2_title} ({f2})")
    print(f"Independent Variable: {feature1_title} ({f1})")
    print(f"\nRegression Equation: y = {intercept:.2f} + ({slope:.2f})x")
    print(f"R-squared: {r_squared:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"\nInterpretation:")
    print(f"- For every 1-unit increase in {f1}, {f2} changes by {slope:.2f}")
    print(f"- R² of {r_squared:.3f} indicates {'strong' if r_squared > 0.5 else 'moderate' if r_squared > 0.2 else 'weak'} correlation")
    print(f"- P-value {'< 0.05 (statistically significant)' if p_value < 0.05 else '> 0.05 (not statistically significant)'}")
    print("=" * 60)
    
    return slope, intercept, r_squared, p_value

def  tree_compare(df, target_col, feature_cols, test_size=0.2, random_state=42):
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error
    
    X_cmp = df[feature_cols].dropna()
    y_cmp = (df.loc[X_cmp.index, target_col] > 0).astype(int)

    X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
        X_cmp, y_cmp, test_size=0.2, random_state=42, stratify=y_cmp
    )

    rf_clf_cmp = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_clf_cmp.fit(X_train_cmp, y_train_cmp)
    y_pred_clf_cmp = rf_clf_cmp.predict(X_test_cmp)
    rmse_clf = root_mean_squared_error(y_test_cmp, y_pred_clf_cmp)

    rf_reg_cmp = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_reg_cmp.fit(X_train_cmp, y_train_cmp)
    y_pred_reg_cmp = rf_reg_cmp.predict(X_test_cmp)
    rmse_reg = root_mean_squared_error(y_test_cmp, y_pred_reg_cmp)

    print(f"RandomForestClassifier RMSE: {rmse_clf:.4f}")
    print(f"RandomForestRegressor RMSE: {rmse_reg:.4f}")
    print("Lower RMSE:", "Classifier" if rmse_clf < rmse_reg else "Regressor")

def regplottter(df, feature1, feature1_title, feature2, feature2_title, feature3, feature3_title):
    """Backward-compatible wrapper for the common misspelling of regplotter."""
    return regplotter(df, feature1, feature1_title, feature2, feature2_title, feature3, feature3_title)

def compare_trees_cal_housing_data(metric_choice="rmse", single_tree_params=None, bagging_params=None, rf_params=None):
    # Compare single DecisionTree, Bagged trees, and RandomForest on California housing
    # - Dataset: California housing (sklearn)
    # - Models:
    #   * Single DecisionTreeRegressor
    #   * BaggingRegressor with DecisionTreeRegressor as estimator (bagging only)
    #   * RandomForestRegressor (bagging + per-split feature subsampling)
    # - For each model we print train/test metrics and feature importances.
    #
    # User-selectable places:
    #  - metric_choice: choose "mse", "rmse", "mae", "r2", or "explained_variance"
    #  - single_tree_params: dict for DecisionTreeRegressor (e.g., {"max_depth": 8})
    #  - bagging_params: dict for BaggingRegressor (e.g., {"n_estimators": 50})
    #  - rf_params: dict for RandomForestRegressor (e.g., {"n_estimators": 100, "max_features": "sqrt"})
    #
    # Note: sklearn APIs vary by version. Modern sklearn uses `estimator=` for BaggingRegressor;
    # older versions used `base_estimator=`. This script uses `estimator=`.

    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

    # ------------------ User-selectable choices ------------------
    # Choose performance metric: "mse", "rmse", "mae", "r2", "explained_variance"
    metric_choice = "rmse"  # change to "mse", "mae", "r2", or "explained_variance"

    # Single decision tree hyperparameters
    single_tree_params = {
        "criterion": "squared_error",  # "squared_error" (MSE) in sklearn >=1.0
        "max_depth": 8,                # set to None to grow fully
        "min_samples_leaf": 1,
        "random_state": 0
    }

    # Bagging parameters (estimator + bagging settings)
    bagging_params = {
        "estimator": DecisionTreeRegressor(criterion="squared_error", max_depth=None, min_samples_leaf=1, random_state=0),
        "n_estimators": 50,            # number of trees in the bag
        "max_samples": 1.0,            # fraction or int (bootstrap sample size)
        "max_features": 1.0,           # fraction or int (features per base estimator)
        "bootstrap": True,
        "bootstrap_features": False,
        "n_jobs": -1,
        "random_state": 0
    }

    # Random forest parameters
    rf_params = {
        "n_estimators": 100,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",        # per-split feature subsampling; change as desired
        "random_state": 0,
        "n_jobs": -1
    }
    # --------------------------------------------------------------

    # Metric wrapper
    def compute_metric(y_true, y_pred, choice="rmse"):
        if choice == "mse":
            return mean_squared_error(y_true, y_pred)
        elif choice == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif choice == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif choice == "r2":
            return r2_score(y_true, y_pred)
        elif choice == "explained_variance":
            return explained_variance_score(y_true, y_pred)
        else:
            raise ValueError("Unknown metric_choice: " + str(choice))

    # Load data
    cal = fetch_california_housing()
    X = pd.DataFrame(cal.data, columns=cal.feature_names)
    y = pd.Series(cal.target, name="MedHouseVal")
    feature_names = list(X.columns)

    # Train/test split (user can adjust test_size and random_state here if desired)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Helper to validate, format, and print feature importances
    def print_feature_importances(importances, names, title, top_n=None, return_df=False):
        if importances is None:
            raise ValueError("importances cannot be None")
        if names is None:
            raise ValueError("names cannot be None")

        names = list(names)
        if len(importances) != len(names):
            raise ValueError(
                f"Length mismatch: importances has {len(importances)} values but names has {len(names)} values."
            )

        s = pd.Series(np.asarray(importances, dtype=float), index=names).sort_values(ascending=False)

        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("top_n must be a positive integer when provided")
            s_to_show = s.head(top_n)
        else:
            s_to_show = s

        total_importance = float(s.sum())
        if np.isclose(total_importance, 0.0):
            out_df = pd.DataFrame({"importance": s_to_show.round(6)})
        else:
            out_df = pd.DataFrame(
                {
                    "importance": s_to_show.round(6),
                    "importance_pct": ((s_to_show / total_importance) * 100).round(2),
                }
            )

        print(title)
        if top_n is not None and top_n < len(s):
            print(f"Showing top {top_n} of {len(s)} features")
        if np.isclose(total_importance, 0.0):
            print("Warning: all feature importances are zero.")
        print(out_df.to_string())
        print()

        if return_df:
            return out_df

    # 1) Single Decision Tree
    dt = DecisionTreeRegressor(**single_tree_params)
    dt.fit(X_train, y_train)
    pred_train_dt = dt.predict(X_train)
    pred_test_dt = dt.predict(X_test)
    metric_train_dt = compute_metric(y_train, pred_train_dt, metric_choice)
    metric_test_dt = compute_metric(y_test, pred_test_dt, metric_choice)

    print("=== Single Decision Tree ===")
    print("Params:", single_tree_params)
    print(f"Train {metric_choice}: {metric_train_dt:.6f}")
    print(f"Test  {metric_choice}: {metric_test_dt:.6f}")
    print_feature_importances(dt.feature_importances_, feature_names, "Feature importances (single tree):")

    # 2) Bagged trees (BaggingRegressor)
    bag = BaggingRegressor(**bagging_params)
    bag.fit(X_train, y_train)
    pred_train_bag = bag.predict(X_train)
    pred_test_bag = bag.predict(X_test)
    metric_train_bag = compute_metric(y_train, pred_train_bag, metric_choice)
    metric_test_bag = compute_metric(y_test, pred_test_bag, metric_choice)

    # Compute averaged feature importances across base estimators if they expose feature_importances_
    base_importances = []
    for est in bag.estimators_:
        # in BaggingRegressor, estimators_ are fitted clones of the provided estimator
        if hasattr(est, "feature_importances_"):
            base_importances.append(est.feature_importances_)
    if len(base_importances) > 0:
        avg_importances = np.mean(base_importances, axis=0)
    else:
        avg_importances = np.zeros(len(feature_names))

    print("=== Bagged Trees (BaggingRegressor) ===")
    print("Params:", {k: v for k, v in bagging_params.items() if k != "estimator"})
    print(f"n_estimators: {bagging_params['n_estimators']}")
    print(f"Train {metric_choice}: {metric_train_bag:.6f}")
    print(f"Test  {metric_choice}: {metric_test_bag:.6f}")
    print_feature_importances(avg_importances, feature_names, "Averaged feature importances (bagged trees):")

    # 3) Random Forest
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    pred_train_rf = rf.predict(X_train)
    pred_test_rf = rf.predict(X_test)
    metric_train_rf = compute_metric(y_train, pred_train_rf, metric_choice)
    metric_test_rf = compute_metric(y_test, pred_test_rf, metric_choice)

    print("=== Random Forest ===")
    print("Params:", rf_params)
    print(f"Train {metric_choice}: {metric_train_rf:.6f}")
    print(f"Test  {metric_choice}: {metric_test_rf:.6f}")
    print_feature_importances(rf.feature_importances_, feature_names, "Feature importances (random forest):")

    summary = pd.DataFrame({
        "model": ["DecisionTree", "BaggedTrees", "RandomForest"],
        "train_" + metric_choice: [metric_train_dt, metric_train_bag, metric_train_rf],
        "test_" + metric_choice: [metric_test_dt, metric_test_bag, metric_test_rf]
    })
    print("=== Summary ===")
    print(summary.to_string(index=False))
    
def compare_random_forest(
    df,
    target_col,
    feature_cols=None,
    test_size=0.2,
    random_state=42,
    threshold=0.0,
    n_estimators=200,
    max_depth=10,
):
    """Compare RandomForestClassifier vs RandomForestRegressor.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target_col : str
        Continuous target column (for example, a delta temperature metric).
    feature_cols : list[str] | None
        Feature columns to use. If None, all columns except target_col are used.
    test_size : float
        Fraction of rows reserved for testing.
    random_state : int
        Seed for reproducibility.
    threshold : float
        Threshold used to create binary labels for the classifier: y > threshold.
    n_estimators : int
        Number of trees for both RF models.
    max_depth : int | None
        Maximum tree depth for both RF models.

    Returns
    -------
    dict
        RMSE values and the better model name.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' is not present in the dataframe.")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"feature_cols contain missing columns: {missing_features}")

    work_df = df[feature_cols + [target_col]].copy()
    work_df = work_df.dropna(subset=[target_col])

    X = work_df[feature_cols]
    y_reg = pd.to_numeric(work_df[target_col], errors="coerce")
    valid_rows = y_reg.notna()
    X = X.loc[valid_rows]
    y_reg = y_reg.loc[valid_rows]

    if len(X) < 10:
        raise ValueError("Not enough valid rows after cleaning to train/test split.")

    y_clf = (y_reg > threshold).astype(int)
    stratify_vec = y_clf if y_clf.nunique() > 1 else None

    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vec,
    )

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    clf_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )

    reg_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )

    clf_model.fit(X_train, y_train_clf)
    y_pred_clf = clf_model.predict(X_test)
    rmse_clf = root_mean_squared_error(y_test_clf, y_pred_clf)

    reg_model.fit(X_train, y_train_reg)
    y_pred_reg = reg_model.predict(X_test)
    rmse_reg = root_mean_squared_error(y_test_reg, y_pred_reg)

    better_model = "Classifier" if rmse_clf < rmse_reg else "Regressor"

    print(f"RandomForestClassifier RMSE: {rmse_clf:.4f}")
    print(f"RandomForestRegressor RMSE: {rmse_reg:.4f}")
    print(f"Lower RMSE: {better_model}")

    return {
        "rmse_classifier": rmse_clf,
        "rmse_regressor": rmse_reg,
        "better_model": better_model,
    }
    
def rft(df, dfname, target_col, feature_cols, depth=7, estimators=7, randomstate=42, numberoffeatures=5, depth_of_trees=4):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    resolved_dfname = dfname or df.attrs.get("name", "dataset")
    df.attrs["name"] = resolved_dfname

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column")

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    fig.suptitle(
        f"Random Forest with {estimators} Trees (max_depth={depth})\n"
        f"Number of tree levels to show: {depth_of_trees}\n"
        f"Dataframe name: {resolved_dfname}",
        fontsize=18,
    )

    # Flatten the axes for easier iteration
    axes = axes.flatten()

    # Prepare data.
    X = df[feature_cols].dropna()
    y = (df.loc[X.index, target_col] > 0).astype(int)

    if X.empty:
        raise ValueError("No rows remain after dropping missing feature values")

    feature_count = numberoffeatures if numberoffeatures is not None else X.shape[1]

    model = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=randomstate)
    model.fit(X, y)

    # Plot each tree in the forest
    trees_to_plot = min(estimators, len(model.estimators_), len(axes) - 1)
    for i, (ax, tree) in enumerate(zip(axes[:trees_to_plot], model.estimators_[:trees_to_plot])):
        plot_tree(tree, 
                filled=True, 
                feature_names=feature_cols,
                class_names=['At/Above Avg', 'Below Avg'],
                rounded=True,
                ax=ax,
                max_depth=depth_of_trees)  
        ax.set_title(f'Tree {i+1}')

    # Remove any unused axes and reserve the last panel for model details.
    for ax in axes[trees_to_plot:-1]:
        ax.axis('off')

    axes[-1].axis('off')

    # Add a text box with model information
    info_text = (
        "Model Information:\n"
        f"- Dataframe: {resolved_dfname}\n"
        f"- Model name: {model.__class__.__name__}\n"
        f"- Number of trees (n_estimators): {estimators}\n"
        f"- Maximum tree depth (max_depth): {depth}\n"
        f"- Number of features (n_features_in_): {feature_count}\n"
        f"- Depth of trees shown: {depth_of_trees}\n"
    )
    axes[-1].text(0.1, 0.5, info_text, fontsize=12, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for the title
    plt.savefig(f'{resolved_dfname}_random_forest.png', dpi=300, bbox_inches='tight')
    # show the plot
    plt.show()
    # close figure to regain memory resources
    plt.close()