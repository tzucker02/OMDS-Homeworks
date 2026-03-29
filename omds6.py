
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
