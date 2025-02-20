from getdata import DataLoader
from makeportfolio import PortfolioOptimizer











# Data
tickers = ['AMZN', 'TSLA', 'AAPL']# ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SQQQ'] #, 'META', 'JPM', 'JNJ', 'V', 'PG', 'DIS']
start_date = '2023-11-30'
end_date = '2024-11-30'

target_return = 0.0015    # цільова дохідність
alpha = 0.05


historical_returns = DataLoader.get_historical_returns(tickers, start_date, end_date)

MakePortfolio = PortfolioOptimizer(historical_returns, target_return, alpha)

result = MakePortfolio.optimize_portfolio_by_Markowitz()
print(result)