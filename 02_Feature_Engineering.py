# merge the features and the responsor on date_time
regions = {}
weather_reg_group = weather_df.groupby("zone")
for c in col:
    left = load_df[c] 
    # the historical weather data is reported every 3 hours, resample so it is hourly  
    right_numerical = weather_reg_group.get_group(c).resample("1H").mean().ffill() 
    right_categorecal = weather_reg_group.get_group(c)[['winddir16Point','weatherDesc']].resample("1H").first().ffill()
    # merge the electirc load dataset and weather history dataset
    comb = pd.merge(left, right_numerical, how = 'inner', left_on = 'date_time', right_on = 'date_time')
    comb = pd.merge(comb, right_categorecal, how = 'inner', left_on = 'date_time', right_on = 'date_time')
    comb.rename(columns={c:'load'},inplace = True).drop(['zone'], axis = 1, inplace =True)
    regions[c] = comb
    print(c, regions[c].shape)

 def feature_engi(df):    
    # create date time features, add year, month, hour, weekday columns and set as categorical variables
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['Hour'] = df.index.hour
    df['weekDay'] = df.index.weekday
    
    # create hourly lag features of 'load'
    df.sort_values('date_time',inplace = True)
    lag_col = ['load'+str(i) for i in range(1,25)]
    for i in range(1,25):
        df[lag_col[i-1]] = df['load'].shift(i)
    
    # create moving average electric load 
    df['load_mv3'] = df['load1'].rolling(3).mean()
    df['load_mv5'] = df['load1'].rolling(5).mean()
    df['load_mv10'] = df['load1'].rolling(10).mean()
    df['load_mv24'] = df['load1'].rolling(24).mean()
    
    # create moving average temperature
    df['tempF_mv3'] =  df['tempF'].rolling(3).mean()    
    df.drop(['tempF'],axis = 1, inplace = True)
    df.dropna(inplace = True)
    
for reg, df in regions.items():
    feature_engi(df)
    print('The dataset shape of {} is {} .'.format(reg, regions[reg].shape))
    
    
