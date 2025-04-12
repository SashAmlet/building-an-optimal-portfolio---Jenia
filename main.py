from getdata import DataLoader
from makeportfolio import PortfolioOptimizer
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

# Data
tickers = ['AAPL', 'MSFT', 'SQQQ']#, 'GOOGL', 'AMZN']
start_date = '2023-11-30'
end_date = '2024-11-30'

target_return = 0.0005   # цільова дохідність
alpha = 0.05

# Analyze the data
historical_returns = DataLoader.get_historical_returns(tickers, start_date, end_date)

MakePortfolio = PortfolioOptimizer(historical_returns, target_return, alpha)

result1 = MakePortfolio.optimize_portfolio_by_Markowitz_1()
result2 = MakePortfolio.optimize_portfolio_by_Markowitz_3()
result3 = MakePortfolio.optimize_portfolio_by_VaR_min()

var1, method1 = MakePortfolio.calculate_VaR(result1)
var2, method2 = MakePortfolio.calculate_VaR(result2)
var3, method3 = MakePortfolio.calculate_VaR(result3)

equal_weights = np.array([1/len(tickers) for _ in range(len(tickers))])
result4 = {"x": equal_weights,
           "fun": MakePortfolio.calculate_volatility(equal_weights)}
var4, method4 = MakePortfolio.calculate_VaR(result4)



# Output results

print("\nMarkowitz-1:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result1['x']])
print('Оптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result1['fun'])
print(f"VaR: {var1:.6f}, Method used: {method1}")

print("\nMarkowitz-3:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result2['x']])
print('Оптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result2['fun'])
print(f"VaR: {var2:.6f}, Method used: {method2}")

print("\nVaR-min:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result3['x']])
print('Оптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result3['fun'])
print(f"VaR: {var3:.6f}, Method used: {method3}")

print("\nStupid-method:")
formatted_weights = ', '.join(['{:.3f}'.format(x) for x in result4['x']])
print('НЕОптимальні ваги активів: [{}]'.format(formatted_weights))
print("Волатильність:", result4['fun'])
print(f"VaR: {var4:.6f}, Method used: {method4}")

MakePortfolio.plot_efficient_frontier()