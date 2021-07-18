mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_HUL = pd.read_csv('HINDUNILVR.csv')
df_TCS = pd.read_csv('TCS.NS.csv')
df_TS = pd.read_csv('TSLA.csv')
df_IC = pd.read_csv('ICICIBANK.csv')
df_CP = pd.read_csv('CIPLA.csv')

df_TCS.dtypes

df_HUL['return'] = 100*((df_HUL['close ']-df_HUL['PREV. CLOSE '])/df_HUL['PREV. CLOSE '])
df_TCS['return'] = 100*((df_TCS['Close']-df_TCS['Adj Close'])/df_TCS['Adj Close'])
df_TS['return'] = 100*((df_TS['Close']-df_TS['Adj Close'])/df_TS['Adj Close'])
df_IC['return'] = 100*((df_IC['close ']-df_IC['PREV. CLOSE '])/df_IC['PREV. CLOSE '])
df_CP['return'] = 100*((df_CP['close ']-df_CP['PREV. CLOSE '])/df_CP['PREV. CLOSE '])

df = pd.DataFrame({'HUL':df_HUL['return'], 'TCS':df_TCS['return'], 'TESLA':df_TS['return'], 'ICICI':df_IC['return'], 'CIPLA':df_CP['return']})

mean_daily_returns = df.mean()
cov_matrix = df.cov()

weights = np.array([0.1,0.2,0.3,0.2,0.2])

portfolio_return = round(np.sum(mean_daily_returns * weights) * 252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
print(f'portfolio return- {portfolio_return}')
print(f'portfolio risk- {portfolio_std_dev}')

num_portfolios = 25000
stocks = ['HUL','TCS','TESLA','ICICI','CIPLA']
results = np.zeros((8,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    weights /= np.sum(weights)
    
    
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i] / results[1,i]
    for j in range(len(weights)):
        results[j+3,i] = weights[j]

results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2],stocks[3],stocks[4]])
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)

print(min_vol_port)

print(max_sharpe_port)
