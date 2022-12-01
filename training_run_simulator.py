"""

Neptune.ai demonstration

"""
from pathlib import Path
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
import shap
import matplotlib.pyplot as plt

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils

import config

pd.set_option("display.max_rows", 10)
pd.set_option("display.precision", 4)

run = neptune.init_run(
    project="aaronwong/breast-cancer-classification",
    api_token=config.NEPTUNE_KEY,
    tags=['classification', 'n=569']
)

params = {
    'early_stopping_rounds': 5,
    'eval_metric': ['aucpr', 'auc'],
    'verbosity': 2
}

# model parameter metadata logging =========================================

run["hyperparameters"] = params

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

X.columns = [c.lower().replace(' ', '_') for c in X.columns]

features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
       'mean_smoothness', 'mean_compactness', 'mean_concavity',
       'mean_concave_points']

filepath = Path.cwd().joinpath('./datasets/features/features.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
feature_df = pd.DataFrame(features)
feature_df.to_csv(filepath)

run["feature_names"].upload('./datasets/features/features.csv')

X = X.loc[:,features]

y = y.loc[:,]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.4, random_state=10
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_valid, y_valid, test_size=0.5, random_state=10
)

train_df = pd.concat([X_train, y_train], axis=1)
valid_df = pd.concat([X_valid, y_valid], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# artifact tracking ================================================

filepath = Path.cwd().joinpath('datasets/train/train_df.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(filepath)

filepath = Path.cwd().joinpath('datasets/valid/valid_df.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
valid_df.to_csv(filepath)

filepath = Path.cwd().joinpath('datasets/test/test_df.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
test_df.to_csv(filepath)

run["dataset_metadata/training_dataset_metadata"].track_files('./datasets/train/train_df.csv')
run["dataset_metadata/validation_dataset_metadata"].track_files('./datasets/valid/valid_df.csv')
run["dataset_metadata/testing_dataset_metadata"].track_files('./datasets/test/test_df.csv')

xgb_model = xgb.XGBClassifier()
xgb_model.set_params(**params)

xgb_model.fit(
    X_train,
    y_train,
    eval_set = [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)],
)

results_log = xgb_model.evals_result()
for train_val, valid_val, test_val in zip(results_log['validation_0']['aucpr'], results_log['validation_1']['aucpr'], results_log['validation_2']['aucpr']):
    run["aucpr_evaluation_metric_logs/training_logs"].log(train_val)
    run["aucpr_evaluation_metric_logs/validation_logs"].log(valid_val)
    run["aucpr_evaluation_metric_logs/testing_logs"].log(test_val)

for train_val, valid_val, test_val in zip(results_log['validation_0']['auc'], results_log['validation_1']['auc'], results_log['validation_2']['auc']):
    run["auc_evaluation_metric_logs/training_logs"].log(train_val)
    run["auc_evaluation_metric_logs/validation_logs"].log(valid_val)
    run["auc_evaluation_metric_logs/testing_logs"].log(test_val)

# uploading dataset (probably a bad idea but just to demonstrate this) ============

run["dataset/training_dataset"].upload('./datasets/train/train_df.csv')
run["dataset/validation_dataset"].upload('datasets/valid/valid_df.csv')
run["dataset/testing_dataset"].upload('./datasets/test/test_df.csv')

# uploading custom metric plots ====================================================

explainer = shap.Explainer(xgb_model)
shap_vals = explainer(X_train)
shap.plots.beeswarm(shap_vals, max_display=10, show=False)
shap_bee_plot = plt.gcf()
run["explainability_plots/training_dataset_SHAP"].upload(shap_bee_plot)

plt.clf()

shap_vals = explainer(X_valid)
shap.plots.beeswarm(shap_vals, max_display=10, show=False)
shap_bee_plot = plt.gcf()
run["explainability_plots/validation_dataset_SHAP"].upload(shap_bee_plot)

plt.clf()

shap_vals = explainer(X_test)
shap.plots.beeswarm(shap_vals, max_display=10, show=False)
shap_bee_plot = plt.gcf()
run["explainability_plots/testing_dataset_SHAP"].upload(shap_bee_plot)

plt_importance = plot_importance(xgb_model, max_num_features=10, importance_type='weight')
run["explainability_plots/xgboost_plot_importance"].upload(plt_importance.figure)

# model performance logging  ==========================================================

# Logging classification summary
# This method creates classifier summary that includes:
# all classifier parameters,
# pickled estimator (model),
# test predictions,
# test predictions probabilities,
# test scores

run["classification_metrics_for_validation_set"] = npt_utils.create_classifier_summary(
    xgb_model, X_train, X_valid, y_train, y_valid
)

run["classification_metrics_for_testing_set"] = npt_utils.create_classifier_summary(
    xgb_model, X_train, X_test, y_train, y_test
)

# pred results validation set ========================================================

y_pred = xgb_model.predict_proba(X_valid, xgb_model.best_ntree_limit)
df_pred = pd.DataFrame(y_pred)

y_pred_non_prob = xgb_model.predict(X_valid)
df_pred_non_prob = pd.DataFrame(y_pred_non_prob)

df_pred_non_prob.rename(columns={0:'y_pred'}, inplace=True)
predictions_valid_df = pd.concat([
    pd.DataFrame(y_pred),
    df_pred_non_prob.reset_index(drop=True),
    y_valid.reset_index(drop=True),
    X_valid.reset_index(drop=True)], axis=1)

predictions_valid_df.rename(columns={'target':'has_cancer (y_true)'}, inplace=True)
predictions_valid_df.to_csv('./predictions/valid/validation_predictions.csv')

run["predictions/validation_set"].upload('./predictions/valid/validation_predictions.csv')

# pred results test set ===============================================================

y_pred = xgb_model.predict_proba(X_test, xgb_model.best_ntree_limit)
df_pred = pd.DataFrame(y_pred)

y_pred_non_prob = xgb_model.predict(X_test)
df_pred_non_prob = pd.DataFrame(y_pred_non_prob)

df_pred_non_prob.rename(columns={0:'y_pred'}, inplace=True)
predictions_test_df = pd.concat([
    pd.DataFrame(y_pred),
    df_pred_non_prob.reset_index(drop=True),
    y_test.reset_index(drop=True),
    X_test.reset_index(drop=True)], axis=1)

predictions_test_df.rename(columns={'target':'has_cancer (y_true)'}, inplace=True)
predictions_test_df.to_csv('./predictions/test/test_predictions.csv')

run["predictions/test_set"].upload('./predictions/test/test_predictions.csv')

# end the training run =================================================================
run.stop()
