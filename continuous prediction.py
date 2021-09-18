# the prediction of next timing relies on the prediction of the previous timing, 
# since the lag load and moving average load features are used
rag_col= ['load1', 'load2', 'load3', 'load4', 'load5', 'load6', 'load7', 'load8',
       'load9', 'load10', 'load11', 'load12', 'load13', 'load14', 'load15',
       'load16', 'load17', 'load18', 'load19', 'load20', 'load21', 'load22',
       'load23', 'load24']
mv_col = ['load_mv3', 'load_mv5', 'load_mv10', 'load_mv24']

# learning by the best model
model_west = best_model_west.fit(X_train,Y_train)

load = list(Y_train.values[-24:])
pred = []
for i in range(fh):
    trans_test = data_prep_pipeline.transform(X_test.iloc[[i]])
    prediction = model_west.predict(trans_test)[0]
    pred.append(prediction)
    load.append(prediction)
    
    if i <= fh-2 :
        for hour, col in enumerate(rag_col):
            X_test.iloc[[i+1]][col] = load[-hour]
        for hours, column in dict(zip([3,5,10,24],mv_col)).items():
            X_test.iloc[[i+1]][column] = sum(load[-hours:])/hours

# plot the result of prediction and true electric load 
plt.figure(figsize=(15,5))
plt.plot(range(168),pred)
plt.plot(range(168),Y_test.values)
plt.legend(['Y_pred','Y_true'],loc='best')
plt.title('West')

