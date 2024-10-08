# model creation - scaled data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.formula.api import ols
import statsmodels.api as sm
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
import seaborn as sns

# import data
all_data = pd.read_excel("all_data_combined 3.xlsx")

print(all_data.head())

# set random state
rs = 32

# specify data for each model
selected_data = all_data[['year','win_pct','Expenses','talent_score', 'class_points','passing_usage','receiving_usage',
             'rushing_usage', 'conference_num', 'avg_rating_All Positions', 'strength_of_schedule',
                          'total_rating_Quarterback', 'total_rating_Receiver',
                          'total_rating_Offensive Line', 'total_rating_Running Back',
                          'total_rating_Linebacker', 'total_rating_Defensive Line',
                          'total_rating_Defensive Back']]

col_names = ['win_pct','Expenses','talent_score', 'class_points','passing_usage','receiving_usage',
             'rushing_usage', 'conference_num', 'avg_rating_All Positions', 'strength_of_schedule',
                          'total_rating_Quarterback', 'total_rating_Receiver',
                          'total_rating_Offensive Line', 'total_rating_Running Back',
                          'total_rating_Linebacker', 'total_rating_Defensive Line',
                          'total_rating_Defensive Back']

# drop na values
selected_data.dropna(inplace = True)
print(len(selected_data))

# drop 2020 data
selected_data[selected_data['year'] != 2020]

# target variable (win_pct)
y = selected_data.win_pct

# predictor variables
x = selected_data.drop(columns = ['win_pct', 'year'])

# scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)

# put data back into data frame
scaled_data = pd.DataFrame(scaled_data, columns = x.columns)

# set up baseline data
baseline_data = scaled_data[['Expenses','passing_usage','receiving_usage',
             'rushing_usage', 'conference_num', 'strength_of_schedule']]

# split data
X_train_knn, X_test_knn, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.3, random_state = rs)
X_train_mlp, X_test_mlp, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.3, random_state = rs)
X_train_rf, X_test_rf, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.3, random_state = rs)
X_train_base, X_test_base, y_train, y_test = train_test_split(baseline_data, y, test_size = 0.3, random_state = rs)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.3, random_state = rs)

# initialize models
baseline = LinearRegression()
knn = KNeighborsRegressor(n_neighbors = 13)
mlp = MLPRegressor(hidden_layer_sizes = (100,), max_iter=500, random_state = rs)
rf = RandomForestRegressor(n_estimators = 7, random_state = rs)

# initialize voting regressor
voting_regressor = VotingRegressor(
    estimators = [('knn', knn), ('mlp', mlp), ('rf', rf)],
    weights = [1,3,1])

# --------------
# evaluate model
# --------------

# fit models
baseline.fit(X_train_base, y_train)
mlp.fit(X_train_mlp, y_train)
rf.fit(X_train_rf, y_train)
knn.fit(X_train_knn, y_train)
voting_regressor.fit(X_train, y_train)

# baseline predictions
base_pred = baseline.predict(X_test_base)

# predict with models
baseline_pred = baseline.predict(X_test_base)
mlp_pred = mlp.predict(X_test_mlp)
rf_pred = rf.predict(X_test_rf)
knn_pred = knn.predict(X_test_knn)

pred_list = [mlp_pred, rf_pred, knn_pred]

# combine predictions (average)
y_pred = voting_regressor.predict(X_test)

# residuals
residuals = y_test - y_pred

# get residuals
mlp_resid = y_test - mlp_pred
rf_resid = y_test - rf_pred
knn_resid = y_test - knn_pred

residuals_list = [mlp_resid, rf_resid, knn_resid]

# convert residuals to dataframe
resid_df = pd.DataFrame({
    'Ensemble': residuals,
    'MLP': mlp_resid,
    'RF': rf_resid,
    'knn': knn_resid
})

bool_df = (resid_df <= .2) & (resid_df >= -.2)
resid_df['within_20'] = bool_df.sum(axis = 1)
resid_df['Actual'] = y_test
resid_df['Ensemble Pred'] = y_pred


# variable importance
rf_importance = rf.feature_importances_
mlp_importance = permutation_importance(mlp, X_test_mlp, y_test, n_repeats = 10, random_state = rs)
knn_importance = permutation_importance(knn, X_test_knn, y_test, n_repeats = 10, random_state = rs) 

# print mlp feature importances
print("MLP Importances: ")
for feature in mlp_importance.importances_mean.argsort()[::-1]:
    print(f"Feature {feature}: {mlp_importance.importances_mean[feature]:.4} +/- {mlp_importance.importances_std[feature]:.4}")

# print mlp feature importances with feature names
features_list = []
imp_list = []
print("\nMLP Importances: ")
for feature in mlp_importance.importances_mean.argsort()[::-1]:
    features_list.append(mlp_data[feature])
    imp_list.append(round(mlp_importance.importances_mean[feature],4))
    print(f"{mlp_data[feature]}: {mlp_importance.importances_mean[feature]:.4}")

# mlp importance dataframe
mlp_importance_df = pd.DataFrame({'Feature': features_list, 'Avg_Importance': imp_list})
print(f"\nNeural Network: \n{mlp_importance_df}\n\n")

# knn importance dataframe
features_list = []
imp_list = []
for feature in knn_importance.importances_mean.argsort()[::-1]:
    features_list.append(knn_data[feature])
    imp_list.append(round(knn_importance.importances_mean[feature],4))

knn_importance_df = pd.DataFrame({'Feature': features_list, 'Avg_Importance': imp_list})
print(f"\nKNN: \n{knn_importance_df}\n\n")


# create dataframce for variable importance
rf_importance_df = pd.DataFrame({'Variable': rf_data, 'Importance': rf_importance})


# sort data frames by importance (descending)
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending = False)

# print summaries
print(f"\nRandom Forest: \n{rf_importance_df}\n\n")

# accuracy metrics (baseline)

# summary (baseline)
r2 = r2_score(y_test, base_pred)
print("Intercept: ", round(baseline.intercept_,3))
print("Coeffecients: ", baseline.coef_)
print()

base_resid = baseline_pred - y_test
mse = mean_squared_error(y_test, baseline_pred)
rmse = math.sqrt(mse)
mae = np.mean(np.abs(base_resid))

within_10_pct = np.sum(np.abs(base_resid) <= .1)
within_15_pct = np.sum(np.abs(base_resid) <= .15)
within_20_pct = np.sum(np.abs(base_resid) <= .2)
print("-" * len("Baseline"))
print("Baseline")
print("-" * len("Baseline"))
print(f'Percent within 10%: {within_10_pct/len(base_resid):.2%}')
print(f'Percent within 15%: {within_15_pct/len(base_resid):.2%}')
print(f'Percent within 20%: {within_20_pct/len(base_resid):.2%}')
print(f'Mean Squared Error: {round(mse,3)}')
print(f'Root Mean Squared Error: {round(rmse,3)}')
print(f'Mean Absolute Error: {round(mae,3)}')
print(f'R-Squared: {r2:.4}')
print("\n\n")

# accuracy metrics (voted)

# r-squared
r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = np.mean(np.abs(residuals))

within_10_pct = np.sum(np.abs(residuals) <= .1)
within_15_pct = np.sum(np.abs(residuals) <= .15)
within_20_pct = np.sum(np.abs(residuals) <= .2)
print("-" * len("Ensemble"))
print("Ensemble")
print("-" * len("Ensemble"))
print(f'Percent within 10%: {within_10_pct/len(residuals):.2%}')
print(f'Percent within 15%: {within_15_pct/len(residuals):.2%}')
print(f'Percent within 20%: {within_20_pct/len(residuals):.2%}')
print(f'Mean Squared Error: {round(mse,3)}')
print(f'Root Mean Squared Error: {round(rmse,3)}')
print(f'Mean Absolute Error: {round(mae,3)}')
print(f'R-Squared: {r2:.4}')
print("\n\n")

# accuracy metrics (individuals)
model_names_list = ["Neural Network", "Random Forest", "K-Nearest Neighbors"]
for model in range(len(pred_list)):
    predicted = pred_list[model]
    resids = residuals_list[model]
    r2 = r2_score(y_test, predicted)
    
    mse = mean_squared_error(y_test, predicted)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(resids))

    within_10_pct = np.sum(np.abs(resids) <= .1)
    within_15_pct = np.sum(np.abs(resids) <= .15)
    within_20_pct = np.sum(np.abs(resids) <= .2)
    print("-" * len(model_names_list[model]))
    print(f'{model_names_list[model]}')
    print("-" * len(model_names_list[model]))
    print(f'Percent within 10%: {within_10_pct/len(resids):.2%}')
    print(f'Percent within 15%: {within_15_pct/len(resids):.2%}')
    print(f'Percent within 20%: {within_20_pct/len(resids):.2%}')
    print(f'Mean Squared Error: {round(mse,3)}')
    print(f'Root Mean Squared Error: {round(rmse,3)}')
    print(f'Mean Absolute Error: {round(mae,3)}')
    print(f'R-Squared: {r2:.4}')
    print("\n\n")



# -----------
# Predictions
# -----------

# read in prediction data
pred_data = pd.read_excel("All Data.xlsx", sheet_name = "Predictive Data")

pred_teams = ["Alabama", "Arkansas", "Michigan", "Texas", "Nebraska", "Florida State",
              "Miami", "Oklahoma", "Notre Dame", "Georgia", "Michigan State", "LSU", "Indiana",
              "Illinois", ]

pred_teams = sorted(pred_teams)

pred_team_df = []

for i in range(len(pred_data)):
    if pred_data["team"][i] in pred_teams:
        pred_team_df.append(pred_data.iloc[i])

pred_team_df = pd.DataFrame(pred_team_df)

print(pred_team_df)

pred_team_df = pred_team_df.drop(columns = ["team", "year", 'avg_rating_Defensive Back', 'avg_rating_Defensive Line', 'avg_rating_Linebacker', 'avg_rating_Offensive Line', 'avg_rating_Quarterback', 'avg_rating_Receiver', 'avg_rating_Running Back', 'avg_rating_Special Teams', 'avg_stars_All Positions', 'avg_stars_Defensive Back', 'avg_stars_Defensive Line', 'avg_stars_Linebacker', 'avg_stars_Offensive Line', 'avg_stars_Quarterback', 'avg_stars_Receiver', 'avg_stars_Running Back', 'avg_stars_Special Teams', 'commits_All Positions', 'commits_Defensive Back', 'commits_Defensive Line', 'commits_Linebacker', 'commits_Offensive Line', 'commits_Quarterback', 'commits_Receiver', 'commits_Running Back', 'commits_Special Teams',
                                            'class_rank', 'classification', 'conference', 'total_rating_All Positions', 'total_rating_Special Teams'])

pred_team_df = pred_team_df[knn_data]
pred_team_df = pd.DataFrame(pred_team_df)

## SCALE DATA
scaled_pred_data = scaler.fit_transform(pred_team_df)
scaled_pred_data = pd.DataFrame(scaled_pred_data, columns = knn_data)

print(scaled_pred_data)

preds_2024 = voting_regressor.predict(scaled_pred_data)

print("\n\n")
for i in range(len(pred_teams)):
    print(f"{pred_teams[i]} - {round(preds_2024[i],3)} ({round(preds_2024[i] * 12, 1)})")

# --------------
# Visualizations
# --------------

x = print(input("\n\nPress any button to continue."))

# predicted, actual, and within_20
sns.scatterplot(data = resid_df, x = 'Ensemble Pred', y = 'Actual', hue = 'within_20', palette = 'Blues_d')
plt.xlabel('Ensemble Prediction')
plt.ylabel('Actual')
plt.title('Figure 9: Ensemble Prediction vs. Actual')
plt.plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
plt.plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
plt.plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
plt.plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
plt.plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower
plt.legend(title = 'Legend', loc = 'upper left')
plt.show()

# baseline model performance
plt.scatter(baseline_pred, y_test, alpha = 0.8, color = 'turquoise')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Baseline Predicted vs. Actual Win Percentage")
plt.plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
plt.plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
plt.plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
plt.plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
plt.plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower
plt.legend(title = 'Legend', loc = 'upper left')
plt.show()

# team comp vs. win pct.
sns.scatterplot(data = all_data, x = 'talent_score', y = 'strength_of_schedule', hue = 'win_pct', palette = 'Blues_d')
plt.xlabel('Talent Score')
plt.ylabel('Strength of Schedule')
plt.title('Figure 5: Talent Score vs. Strength of Schedule colored by Win Percentage')
plt.show()

# team comp vs. win pct
sns.scatterplot(data = all_data, x = 'talent_score', y = 'win_pct')
plt.xlabel('Talent Score')
plt.ylabel('Win Percentage')
plt.title('Talent Score vs. Win Percentage')
plt.show()

# predicted vs. actual
plt.scatter(y_pred, y_test, alpha = 0.8, color = 'skyblue')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Predicted vs. Actual Win Percentage")
plt.plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
plt.plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
plt.plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
plt.plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
plt.plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower
plt.legend(title = 'Legend', loc = 'upper left')
plt.show()

# predicted vs. actual for all models
fig, axs = plt.subplots(2,2)

axs[0,0].scatter(y_pred, y_test, color = 'lightskyblue')
axs[0,0].set_title("Ensemble")
axs[0,0].set_xlabel("Predicted")
axs[0,0].set_ylabel("Actual")
axs[0,0].plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
axs[0,0].plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
axs[0,0].plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
axs[0,0].plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
axs[0,0].plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower

axs[0,1].scatter(mlp_pred, y_test, color = 'slategrey')
axs[0,1].set_title("Neural Network")
axs[0,1].set_xlabel("Predicted")
axs[0,1].set_ylabel("Actual")
axs[0,1].plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
axs[0,1].plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
axs[0,1].plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
axs[0,1].plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
axs[0,1].plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower

axs[1,0].scatter(rf_pred, y_test, color = 'mediumseagreen')
axs[1,0].set_title("Random Forest")
axs[1,0].set_xlabel("Predicted")
axs[1,0].set_ylabel("Actual")
axs[1,0].plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
axs[1,0].plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
axs[1,0].plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
axs[1,0].plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
axs[1,0].plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower

axs[1,1].scatter(knn_pred, y_test, color = 'wheat')
axs[1,1].set_title("K-Nearest Neighbors")
axs[1,1].set_xlabel("Predicted")
axs[1,1].set_ylabel("Actual")
axs[1,1].plot([0,1], [0,1], color='red', label = 'Perfect fit') # perfect fit
axs[1,1].plot([0, 0.9], [0.1, 1.0], color = 'darkorange', linestyle = '--', label = '+/-10%') # 10% higher
axs[1,1].plot([0.1, 1.0], [0, 0.9], color = 'darkorange', linestyle = '--') # 10% lower
axs[1,1].plot([0, 0.8], [0.2, 1.0], color = 'tab:olive', linestyle = '--', label = '+/-20%') # 20% higher
axs[1,1].plot([0.2, 1.0], [0, 0.8], color = 'tab:olive', linestyle = '--') # 20% lower

plt.tight_layout()
plt.show()

# residuals vs. predicted
plt.scatter(y_pred, residuals, alpha = 0.5)
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.axhline(y = 0, color = 'red', label = 'Perfect fit')
plt.axhline(y = 0.1, color = 'darkorange', linestyle = '--', label = '+/-10%')
plt.axhline(y = -0.1, color = 'darkorange', linestyle = '--')
plt.axhline(y = 0.2, color = 'tab:olive', linestyle = '--', label = '+/-20%')
plt.axhline(y = -0.2, color = 'tab:olive', linestyle = '--')
plt.legend(title = 'Legend', loc = 'lower left')
plt.show()

# residuals histogram
sns.histplot(residuals)
plt.xlabel("Residual")
plt.title("Figure 8: Histogram of Residuals")
plt.show()


# view decision tree
##plt.figure(figsize=(20,10))
##plot_tree(mlp, feature_names = x.columns, filled = True)
##plt.show()
