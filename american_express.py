from easydict import EasyDict
import random
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
configuration = EasyDict({
    "input_dir":'/kaggle/working/',
    "seed":40,
    "n_folds":5,
    "target":'target',
    "boosting_type":'dart',
    "metric":'binary_logloss',
    "cat_features":[
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
})

# code for Seeding part
random.seed(configuration.seed)
np.random.seed(configuration.seed)
 
# Metric for Amex 
def amex_metric(y_true, y_pred):
    # Create labels array
    labels = np.array([y_true, y_pred]).T
    # Sort labels based on y_pred in descending order
    labels = labels[np.argsort(-y_pred)]
    # This is done to give more importance to the negative class
    weights = np.where(labels[:, 0] == 0, 20, 1)
    # Get top 4% of the labels based on weight
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]
    for i in [1, 0]:
        labels = np.array([y_true, y_pred]).T
        labels = labels[np.argsort(-y_pred if i else -y_true)]
        # Assign weight of 20 for negative class and 1 for positive class
        weight = np.where(labels[:, 0] == 0, 20, 1)
        # weight_random is the cumulative sum of the weight divided by the sum of the weight
        weight_random = np.cumsum(weight / np.sum(weight))
        # lorentz is the cumulative sum of the positive class
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    # The metric is the average of the gini score and the top 4% of the labels
    return 0.5 * (gini[1] / gini[0] + top_four)

# Metric for LightGBM
def light_gbm_amex_metric(y_pred, y_true):
    # Get true labels
    y_true = y_true.get_label()
    # Calculate amex_metric
    metric_val = amex_metric(y_true, y_pred)
    return 'amex_metric', metric_val, True

# Metric for XGBoost
# Preprocessing data
def get_diff(data, num_features):
    """
    Function to calculate the difference of numeric features inside a dataframe and group by 'customer_ID'
    """
    # Initialize lists to store differences and customer IDs
    dataframe1 = []
    customer_ids = []

    # Iterate over groups of dataframe, grouped by 'customer_ID'
    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        # Calculate the differences of num_features
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        dataframe1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate the differences and customer IDs
    dataframe1 = np.concatenate(dataframe1, axis=0)
    # Transform to dataframe
    dataframe1 = pd.DataFrame(dataframe1, columns=[col + "_diff1" for col in df[num_features].columns])
    # Add customer id
    dataframe1["customer_ID"] = customer_ids
    return dataframe1


def read_pre_process_data():
    """
    Function to read, preprocess, and aggregate data
    """
    # Read train data
    train = pd.read_csv("/kaggle/input/amex-default-prediction/train_data.csv")
    # Identify categorical and numerical features
    features = train.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    # Identify numerical features   
    num_features = [col for col in features if col not in cat_features]
    print("Starting training feature engineering...")
    # Calculate the difference of numeric features inside a dataframe and group by 'customer_ID'
    train_num_agg = train.groupby("customer_ID")[num_features].agg(["mean", "std", "min", "max", "last"])
    train_num_agg.columns = ["_".join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace=True)
    # Aggregate categorical features by customer_ID
    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(["count", "last", "nunique"])
    train_cat_agg.columns = ["_".join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace=True)
    # Read train labels
    train_labels = pd.read_csv("/kaggle/input/amex-default-prediction/train_labels.csv")
    # Converting float64 columns to float32
    # This is done to reduce the memory usage
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == "float64"].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    # Converting int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == "int64"].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)
    # Calculate differences of numeric features by customer_ID
    train_diff = get_diff(train, num_features)
    # Merge the aggregated features, differences, and labels
    train = train_num_agg.merge(
        train_cat_agg, how="inner", on="customer_ID"
    ).merge(train_diff, how="inner", on="customer_ID").merge(
        train_labels, how="inner", on="customer_ID"
    )
    # Read test data
    
    test = pd.read_parquet("/kaggle/input/amex-default-prediction/test_data.csv")
    print("Starting test feature engineering...")
    # Perform the same operations on the test data
    test_num_agg = test.groupby("customer_ID")[num_features].agg(["mean", "std", "min", "max", "last"])
    test_num_agg.columns = ["_".join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace=True)
    test_cat_agg = test.groupby("customer_ID")[cat_features].agg(["count", "last", "nunique"])
    test_cat_agg.columns = ["_".join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace=True)
    test_diff = get_diff(test, num_features)
    test = test_num_agg.merge(test_cat_agg, how="inner", on="customer_ID").merge(
        test_diff, how="inner", on="customer_ID"
    )
    # Return the preprocessed data
    return train, test

# Read and preprocess data
train,test = read_pre_process_data()

# Label encode categorical features
def train_and_evaluate(train, test):
    # Label encode categorical features
    cat_cols = configuration.cat_features
    cat_cols = [f"{col}_last" for col in cat_cols]
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    # Get the difference between last and mean
    float_cols = train.select_dtypes(include=['float']).columns
    float_cols = [col for col in float_cols if 'last' in col]
    # Round the float columns
    train[float_cols] = train[float_cols].round(2)
    test[float_cols] = test[float_cols].round(2)

    # Get the difference between last and mean
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    # Get the difference between last and mean
    for col in num_cols:
        train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
        test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
    # Transform float64 and float32 to float16
    # This will reduce the memory usage
    float_cols = train.select_dtypes(include=['float']).columns

    # Round the float columns
    train[float_cols] = train[float_cols].astype(np.float16)
    test[float_cols] = test[float_cols].astype(np.float16)
    # Get feature list
    # We will not use customer_ID as a feature
    features = [col for col in train.columns if col not in ['customer_ID', configuration.target]]
    # Define model parameters
    # We will use the same parameters for all models
    params = {
        'objective': 'binary',
        'metric': configuration.metric,
        'boosting': configuration.boosting_type,
        'seed': configuration.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }

    # Create a numpy array to store test predictions
    test_predictions = np.zeros(len(test))
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train))
    from sklearn.model_selection import StratifiedKFold, train_test_split
    import lightgbm as lightgbm
    kfold = StratifiedKFold(n_splits=configuration.n_folds, shuffle=True, random_state=configuration.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[configuration.target])):
        # Print fold number
        print(f'\nTraining fold {fold} with {len(features)} features...')
        
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[configuration.target].iloc[trn_ind], train[configuration.target].iloc[val_ind]
        lightgbm_train = lightgbm.Dataset(x_train, y_train, categorical_feature=cat_cols)
        lightgbm_val = lightgbm.Dataset(x_val, y_val, categorical_feature=cat_cols)
        model = lightgbm.train(params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_val],
                               valid_names=['train', 'val'], num_boost_round=1000,
                               early_stopping_rounds=50, verbose_eval=50,
                               feval=light_gbm_amex_metric)
        oof_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(test[features]) / configuration.n_folds
        score = amex_metric(y_val, model.predict(x_val))

        # Print the score
    score = amex_metric(train[configuration.target], oof_predictions)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/kaggle/working/submission.csv', index=False)

# Train and evaluate the model we created
train_and_evaluate(train, test)
