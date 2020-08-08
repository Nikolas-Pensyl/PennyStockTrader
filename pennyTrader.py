from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import scipy.stats as scpy
import yfinance as yf
import pandas as pd
import numpy as np

symbols = ['GSUM','MDGS','FLDM','AMTX','SPI','MIST','MVIS','BHAT','SLCA','AKRX','NHLD','LIVX','RTTR','TKAT','NRT','OGEN','INSE','WHLM','MTP','AGRX','YTRA','MYO','BIMI','OPTT','SGBX','SNDL','HSDT','GMBL','CWBR','FNGD','MLND','CJJD','OCGN','GENE','SIEN','ACHV','FRAN','PME','AQMS','GALT','FAT','GTEC','RIBT','OFS','NURO','DWSN','XELB','GVP','WKHS','WKHS','WKHS','PRCP','ECT','JBR','EVOL','PEI','KTP','AMPY','KMPH','GROW','TRIL','ATHX',]
data = yf.download(symbols, period = "1y",interval = '1d' )
data = data.fillna(0)
longs = []
shorts = []
for i in range(len(data)-66):
    print(str(100*i/(len(data)-66))+'% complete')
    opens = []
    closes = []
    decisions = []
    
    for symbol in symbols:
        
        #work in a pclose
        X = data['Open',symbol][i+1:i+64].to_numpy().reshape(-1,1)
        y = data['Close',symbol][i+1:i+64]
        #Split data into train and test
        train_X, val_X, train_y, val_y = train_test_split(X, y)
    
        #Create model and predictions
        model = linear_model.LinearRegression()
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        
        #Compute stock 95% confidence interval
        l1 = preds_val - val_y
        mean = np.average(l1)
        stdev = np.std(l1)
        predict = data['Open',symbol][i+65].reshape(1,-1)
        pred_increase = model.predict(predict)- data['Open',symbol][i+65]
        #Compute long/short with confidence 
        zscore = 3
        while True:
            lowf = mean - zscore*stdev + pred_increase
            highf = mean + zscore*stdev + pred_increase
            
            if lowf > 0 and highf > 0:
                prob = scpy.norm.sf(abs(zscore))*2
                prob = 1 - prob
                decision = round(prob*100,2)
                break
            
            elif lowf < 0 and highf < 0:
                prob = scpy.norm.sf(abs(zscore))*2
                prob = 1 - prob
                decision = round(prob*100,2) *-1
                break 
            
            else:
                zscore -= .001
        
        #Add data to lists
        opens.append(data['Open',symbol][i+65])
        closes.append(data['Close',symbol][i+65])
        decisions.append(decision)
        
    #Turn lists into a Dataframe 
    df = pd.DataFrame({'ticker': symbols,'date': data.index[i+65],'open': opens,'decision': decisions,'close':closes})
    df_sorted = df.sort_values(by=['decision']).reset_index()
    longs.append([df_sorted['date'][len(df_sorted)-1],df_sorted['ticker'][len(df_sorted)-1],df_sorted['open'][len(df_sorted)-1],df_sorted['decision'][len(df_sorted)-1],df_sorted['close'][len(df_sorted)-1]])
    shorts.append([df_sorted['date'][0],df_sorted['ticker'][0],df_sorted['open'][0],df_sorted['decision'][0],df_sorted['close'][0]])

#Makes dataframes for performance over 9 months 
longperformance = pd.DataFrame(longs, columns=['date','ticker','open','decision','close'])
shortperformance = pd.DataFrame(shorts, columns=['date','ticker','open','decision','close'])
longprofit = lambda row: row['close'] - row['open']
shortprofit = lambda row: row['open'] - row['close'] 
longperformance['profit'] = longperformance.apply(longprofit, axis = 1)
shortperformance['profit'] = shortperformance.apply(shortprofit, axis = 1)
print(longperformance['profit'].sum())
print(shortperformance['profit'].sum())
longperformance.to_csv('longperformance.csv')
shortperformance.to_csv('shortperformance.csv')
