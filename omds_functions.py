import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import importlib


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

    # Helper to format and print feature importances
    def print_feature_importances(importances, names, title):
        s = pd.Series(importances, index=names).sort_values(ascending=False)
        print(title)
        print(s.to_string())
        print()

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

    # Summary table
    summary = pd.DataFrame({
        "model": ["DecisionTree", "BaggedTrees", "RandomForest"],
        "train_" + metric_choice: [metric_train_dt, metric_train_bag, metric_train_rf],
        "test_" + metric_choice: [metric_test_dt, metric_test_bag, metric_test_rf]
    })
    print("=== Summary ===")
    print(summary.to_string(index=False))

    # End of script