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

# %%
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
        hue=feature3[0],
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


def regplottter(df, feature1, feature1_title, feature2, feature2_title, feature3, feature3_title):
    """Backward-compatible wrapper for the common misspelling of regplotter."""
    return regplotter(df, feature1, feature1_title, feature2, feature2_title, feature3, feature3_title)