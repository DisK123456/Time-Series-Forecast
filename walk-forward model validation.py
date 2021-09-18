from xgboost import XGBRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space={'gamma': hp.uniform ('gamma', 1, 9),
       'reg_alpha' : hp.quniform('reg_alpha', 40, 200, 1),
       'reg_lambda' : hp.uniform('reg_lambda', 0, 2),
       'learning_rate' : hp.choice('learning_rate',[0.001, 0.01, 0.1, 0.3])
       }

def parameter_tuning(space):
    
    model= XGBRegressor(gamma=space['gamma'],
                         reg_alpha=int(space['reg_alpha']), 
                         reg_lambda = space['reg_lambda'],
                         learning_rate = space['learning_rate'],                        
                         n_estimators=180,max_depth=6, 
                         base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bynode=1, colsample_bytree=0.64,
                         gpu_id=-1, importance_type='gain',
                         interaction_constraints='', 
                         max_delta_step=0, min_child_weight=6.0, missing=None,
                         monotone_constraints='()', n_jobs=8,
                         num_parallel_tree=1, random_state=0, 
                         scale_pos_weight=1, subsample=1, tree_method='exact',
                         validate_parameters=1, verbosity=None)
    
    # walk-forward cross validation
    RMSE = []
    for i in range(len(val)):
        evaluation = [(X_val[i], Y_val[i])]
        model.fit(X_train[i], Y_train[i])
        pred = model.predict(X_val[i])
        rmse = np.sqrt(mean_squared_error(Y_val[i], pred))
        RMSE.append(rmse)  
    
    return {'loss': sum(RMSE)/len(RMSE), 'status': STATUS_OK, 'model': model} 
  
# choose the best model by fmin function
trials = Trials()
best_model = fmin(fn = parameter_tuning, 
                  space = space,
                  algo = tpe.suggest,
                  max_evals = 20,
                  trials = trials )

