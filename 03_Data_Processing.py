from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# fh is the number of hours need to be forecasted, in this competition, it is 168 hours (7days)
fh = 168
# train/test splitting for final testing
# train_test_regeions = {'regions': [train,test]} 

train_test_regions = {}
for reg, df in regions.items():
    test = df.iloc[-1*fh:,:]
    train = df.iloc[:-1*fh:,:]
    train_test_regions[reg] = [train,test]
    
# train / validation set splitting(3 folds) for parameter tuning and best model selection
# size of train and val set : 90% vs 10% (approximately 3000 cases)
# train_val_regeions = {'regions': [[train1,val1],[train2,val2],[train3,val3]]}
val_size = 3000
train_val_regions = {}
for reg, df in train_test_regions.items():
    data = []
    n = len(df[0])
    for i in range(1,4):
        val = df[0].iloc[-1*val_size*i:n-val_size*(i-1),:]
        train = df[0].iloc[:-1*val_size*i,:]
        data.append([train,val])
    train_val_regions[reg] = data

# data processing pipeline: standard scale numerical columns and one-hot encode categorical columns
cat = [['S', 'ESE', 'SSE', 'SE', 'N', 'SSW', 'NE', 'NNE', 'E', 'ENE','WSW', 'SW', 'NNW', 'W', 'WNW', 'NW'],
      ['Clear', 'Patchy rain possible', 'Partly cloudy', 'Sunny',
       'Patchy light drizzle', 'Moderate or heavy rain shower',
       'Light rain shower', 'Moderate rain', 'Heavy rain at times',
       'Patchy light rain', 'Mist', 'Light drizzle', 'Light rain',
       'Cloudy', 'Thundery outbreaks possible', 'Fog',
       'Moderate rain at times', 'Torrential rain shower', 'Overcast',
       'Patchy light rain with thunder', 'Heavy rain',
       'Moderate or heavy rain with thunder', 'Blowing snow', 'Blizzard',
       'Heavy snow', 'Moderate snow', 'Patchy heavy snow', 'Ice pellets',
       'Light sleet', 'Heavy freezing drizzle', 'Light freezing rain',
       'Light showers of ice pellets', 'Moderate or heavy freezing rain',
       'Patchy moderate snow', 'Patchy sleet possible',
       'Moderate or heavy showers of ice pellets', 'Patchy light snow',
       'Patchy freezing drizzle possible', 'Light snow',
       'Moderate or heavy sleet', 'Patchy snow possible', 'Freezing fog',
       'Light sleet showers', 'Freezing drizzle',
       'Moderate or heavy snow showers', 'Light snow showers'],
      [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021],
      [1,2,3,4,5,6,7,8,9,10,11,12],
      [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
      [0,1,2,3,4,5,6]]

cat_col = ['winddir16Point', 'weatherDesc','year', 'month', 'Hour', 'weekDay']
num_col = ['windspeedMiles', 'visibilityMiles','winddirDegree', 'precipMM',
           'humidity', 'visibility', 'pressure', 'cloudcover', 'uvIndex', 
           'load1', 'load2', 'load3', 'load4', 'load5', 'load6', 'load7', 'load8', 
           'load9', 'load10', 'load11', 'load12','load13', 'load14', 'load15', 'load16', 
           'load17', 'load18', 'load19', 'load20', 'load21', 'load22', 'load23', 'load24', 
           'load_mv3','load_mv5', 'load_mv10', 'load_mv24', 'tempF_mv3']

col_preprocessing = ColumnTransformer([('cat_col_preprocessing', OneHotEncoder(sparse=False, categories = cat, handle_unknown='ignore'), cat_col),
                                       ('num_col_preprocessing', StandardScaler(), num_col)], 
                                       remainder='passthrough')
data_prep_pipeline = Pipeline([('col_preprocessing', col_preprocessing)], verbose=True)

# use the data of west to roughly search for the good models and range of the parameters
train = [i[0] for i in train_val_regions['West']]
val = [i[1] for i in train_val_regions['West']]
X_train = [data_prep_pipeline.fit_transform(i.drop(['load'],axis=1)) for i in train]
Y_train = [ i['load'] for i in train]
X_val =  [data_prep_pipeline.transform(i.drop(['load'],axis=1)) for i in val]
Y_val = [ i['load'] for i in val]
