# pip install git+https://github.com/hyperopt/hyperopt-sklearn
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing, xgboost_regression

# initially search for the best model
model = HyperoptEstimator(regressor=any_regressor('reg'), 
                          preprocessing=any_preprocessing('pre'), 
                          loss_fn=mean_squared_error, 
                          algo=tpe.suggest, 
                          max_evals=200, 
                          trial_timeout=120)

model.fit(X_train[0], Y_train[0])
mse = model.score(X_val[0], Y_val[0])
print("MSE: %.3f" % mse)
print(model.best_model())

# Initial results show XGBoosterRgressor has promising performance.
