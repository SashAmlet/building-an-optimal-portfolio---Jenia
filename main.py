from getdata import DataLoader
from makeportfolio import PortfolioOptimizer
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Data
tickers = ['AMZN', 'TSLA', 'AAPL']# ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SQQQ'] #, 'META', 'JPM', 'JNJ', 'V', 'PG', 'DIS']
start_date = '2023-11-30'
end_date = '2024-11-30'

target_return = 0.0015    # цільова дохідність
alpha = 0.05

# Analyze the data
historical_returns = DataLoader.get_historical_returns(tickers, start_date, end_date)

MakePortfolio = PortfolioOptimizer(historical_returns, target_return, alpha)

result = MakePortfolio.optimize_portfolio_by_Markowitz()

var1_zero, var1_mean, var2_zero, var2_mean = MakePortfolio.calculate_VaR(result)



# Output results
print("VaR-zero історичний:", var1_zero)
print("VaR-mean історичний:", var1_mean)
print("VaR-zero параметричний:", var2_zero)
print("VaR-mean параметричний:", var2_mean)

print("Оптимальні ваги активів:", result.x)
print("Волатильність:", result.fun)

MakePortfolio.plot_efficient_frontier()