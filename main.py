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

result1 = MakePortfolio.optimize_portfolio_by_Markowitz()
result2 = MakePortfolio.optimize_portfolio_by_VaR_1()

var1_zero, var1_mean, var2_zero, var2_mean = MakePortfolio.calculate_VaR(result1)



# Output results
print("VaR-zero історичний:", var1_zero)
print("VaR-mean історичний:", var1_mean)
print("VaR-zero параметричний:", var2_zero)
print("VaR-mean параметричний:", var2_mean)

print("Markowitz:")
print("Оптимальні ваги активів:", result1['x'])
print("Волатильність:", result1['fun'])

print("VaR1:")
print("Оптимальні ваги активів:", result2['x'])
print("Волатильність:", result2['fun'])


# MakePortfolio.plot_efficient_frontier()