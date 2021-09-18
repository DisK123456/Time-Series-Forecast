# Time-Series-Forecast-XGBoost
#### Data Souce: https://www.kaggle.com/c/wids-texas-datathon-2021/data
Project information: the target of this project is to forecast the hourly electic load of eight weather zones in Texas in the next 7 days. The dataset is historical load data from the Electric Reliability Council of Texas (ERCOT) and tri-hourly weather data in major cities cross ECROT weather zones.
# Highlight of this sharing:
- Multivariate time series prediction
- Model selection by hp_sklearn
- Hyperparameter tuning by hyperopt
- Continuous prediction in XGB
# List of python files:
- Data_Exploration.py : explore the patern of distribution and correlation 
- Feature_Engineering.py : add lag features, rolling average features and other related features, drop highly correlated features
- Data_Processing.py: one-hot-encode and standarize 
- Model_Selection.py : use hp-sklearn package to initially search for the best model, and use hyperopt package to tune parameters
- Walk-forward_Cross_Validation.py : walk-forward cross validation strategy to preserve the temporal order of observations
- Continuous_Prediction.py : use the prediction of current timing to predict next timing because the lag and rolling average features are used

# Plot of Result
![Prediction of West region](https://github.com/DisK123456/Time-Series-Forecast/blob/main/prediction_of_west.png?raw=true)
