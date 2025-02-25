from getdata import DataLoader
from makeportfolio import PortfolioOptimizer
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SQQQ'] #, 'META', 'JPM', 'JNJ', 'V', 'PG', 'DIS']
start_date = '2023-11-30'
end_date = '2024-11-30'

target_return = 0.0015    # цільова дохідність
alpha = 0.05

# Analyze the data
historical_returns = DataLoader.get_historical_returns(tickers, start_date, end_date)

MakePortfolio = PortfolioOptimizer(historical_returns, target_return, alpha)

result1 = MakePortfolio.optimize_portfolio_by_Markowitz_1()
result2 = MakePortfolio.optimize_portfolio_by_Markowitz_3()
result3 = MakePortfolio.optimize_portfolio_by_VaR_min()

var1_zero, var1_mean, var2_zero, var2_mean = MakePortfolio.calculate_VaR(result1)



# Output results
print("VaR-zero історичний:", var1_zero)
print("VaR-mean історичний:", var1_mean)
print("VaR-zero параметричний:", var2_zero)
print("VaR-mean параметричний:", var2_mean)


print("Markowitz-1:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result1['x']])
print('Оптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result1['fun'])

print("Markowitz-3:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result2['x']])
print('Оптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result2['fun'])

print("VaR-min:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result3['x']])
print('Оптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result3['fun'])

MakePortfolio.plot_efficient_frontier()