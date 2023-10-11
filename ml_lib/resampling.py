# resampling.py

### Resample Datasets to deal with Imbalances (Optional)

def validate_inputs(datasets: Dict[str, pd.DataFrame], resample_label: str, resample_strategy: str) -> None:
    if not isinstance(datasets, dict):
        raise ValueError("Input datasets should be a dictionary.")
    if resample_strategy not in ["upsample", "downsample", "combined"]:
        raise ValueError("Invalid resample_strategy. Choose from 'upsample', 'downsample', or 'combined'.")
    for key, df in datasets.items():
        if resample_label not in df.columns:
            raise ValueError(f"'{resample_label}' is not a valid column in the {key} dataset.")

def target_count_for_strategy(label_counts: pd.Series, strategy: str) -> int:
    if strategy == "downsample":
        return label_counts.min()
    elif strategy == "upsample":
        return label_counts.max()
    return int(label_counts.median())

def iterative_resampling(df: pd.DataFrame, resample_strategy: str, resample_label: str) -> pd.DataFrame:
    label_counts = df[resample_label].apply(tuple).value_counts()
    target_count = target_count_for_strategy(label_counts, resample_strategy)
    subsets = [
        resample(
            df[df[resample_label].apply(tuple) == unique_label],
            replace=(label_counts[unique_label] < target_count),
            n_samples=target_count
        )
        for unique_label in label_counts.keys()
    ]
    return pd.concat(subsets).sample(frac=1).reset_index(drop=True)

def resample_datasets(datasets: Dict[str, pd.DataFrame], resample_label='Multi_Output_Labels', resample_strategy="downsample") -> Dict[str, pd.DataFrame]:
    validate_inputs(datasets, resample_label, resample_strategy)
    
    int32_columns = [col for col, dtype in datasets.get('train', pd.DataFrame()).dtypes.items() if dtype == 'int32']
    
    def process_dataset(key: str, df: pd.DataFrame) -> pd.DataFrame:
        if key != 'train':
            return df
        resampled_data = iterative_resampling(df, resample_strategy, resample_label)
        for col in int32_columns:
            resampled_data[col] = resampled_data[col].astype('int32')
        return resampled_data
    
    return {key: process_dataset(key, df) for key, df in datasets.items()}


