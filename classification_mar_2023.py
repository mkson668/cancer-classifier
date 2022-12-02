"""

Neptune.ai demonstration

"""
from pathlib import Path
import pandas as pd
from xgboost import plot_importance
from sklearn.datasets import load_breast_cancer
# from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
import shap
import matplotlib.pyplot as plt
import neptune.new as neptune
import config
import pickle

pd.set_option("display.max_rows", 10)
pd.set_option("display.precision", 4)

run_BRST_CLSFR_4 = neptune.init_run(
    project="aaronwong/breast-cancer-classification",
    with_id="BRST-75",
    api_token=config.NEPTUNE_KEY,
)

run_BRST_CLSFR_4["classification_metrics_for_testing_set/pickled_model"].download()

run = neptune.init_run(
    project="aaronwong/breast-cancer-classification",
    api_token=config.NEPTUNE_KEY,
    tags=['classification', 'n=250', 'production-Feb-2023']
)

# model parameter metadata logging =========================================


X, y = load_breast_cancer(return_X_y=True, as_frame=True)

X.columns = [c.lower().replace(' ', '_') for c in X.columns]

features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
       'mean_smoothness', 'mean_compactness', 'mean_concavity',
       'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
       'radius_error', 'texture_error']

filepath = Path.cwd().joinpath('./datasets/features/features.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
feature_df = pd.DataFrame(features)
feature_df.to_csv(filepath)

run["feature_names"].upload('./datasets/features/features.csv')

X = X.loc[:400,features]

y = y.loc[:400,]

# artifact tracking ================================================

filepath = Path.cwd().joinpath('datasets/production_dataset/production_df.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
X.to_csv(filepath)

run["dataset_metadata/training_dataset_metadata"].track_files('./datasets/production_dataset/production_df.csv')

# to track data usage using project metadata (first we need to upload to project metadata first btw)
# proj = neptune.init_project(name="aaronwong/breast-cancer-classification", api_token=config.NEPTUNE_KEY)
# run["dataset_metadata/project_dataset_ver"] = proj['datasets/v0.2'].fetch()

run_BRST_CLSFR_4['classification_metrics_for_testing_set/pickled_model'].download('./models')

xgb_model = pickle.load(open('./pickled_model.pkl', 'rb'))

# uploading dataset (probably a bad idea but just to demonstrate this) ============

run["dataset/production_dataset"].upload('./datasets/production_dataset/production_df.csv')

# uploading custom metric plots ====================================================

explainer = shap.Explainer(xgb_model)
shap_vals = explainer(X)
shap.plots.beeswarm(shap_vals, max_display=10, show=False)
shap_bee_plot = plt.gcf()
run["explainability_plots/production_dataset_SHAP"].upload(shap_bee_plot)

plt.clf()

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

# pred results production set ========================================================

y_pred = xgb_model.predict_proba(X, xgb_model.best_ntree_limit)
df_pred = pd.DataFrame(y_pred)

y_pred_non_prob = xgb_model.predict(X)
df_pred_non_prob = pd.DataFrame(y_pred_non_prob)

df_pred_non_prob.rename(columns={0:'y_pred'}, inplace=True)
predictions_valid_df = pd.concat([
    pd.DataFrame(y_pred),
    df_pred_non_prob.reset_index(drop=True),
    X.reset_index(drop=True)], axis=1)

predictions_valid_df.rename(columns={'target':'has_cancer (y_true)'}, inplace=True)
predictions_valid_df.to_csv('./predictions/production/production_predictions.csv')

run["predictions/production_set"].upload('./predictions/production/production_predictions.csv')

# terminate runs =================================================================
run.stop()
run_BRST_CLSFR_4.stop()
