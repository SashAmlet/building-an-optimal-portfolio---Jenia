import numpy as np
from scipy.optimize import minimize
import sys

class PortfolioOptimizer:
    def __init__(self, historical_returns, target_return, alpha):
        self.historical_returns = historical_returns
        self.target_return = target_return
        self.alpha = alpha

        self.mean_returns = np.mean(historical_returns, axis=0)  # очікувані дохідності
        
        if all(target_return > element for element in self.mean_returns):
            print("ERROR")
            sys.exit()

        self.log_historical_returns, self.log_target_return, self.log_mean_returns = self.logarithmization()


    
    def logarithmization(self):
        log_returns = np.log(self.historical_returns + 1)
        log_target_return = np.log(self.target_return + 1)
        log_mean_returns = np.log(self.mean_returns + 1)

        return log_returns, log_target_return, log_mean_returns

    def optimize_portfolio_by_Markowitz(self, log=True):
        # Логарифмізація
        if log:
            mean_returns = self.log_mean_returns
            cov_matrix = np.cov(self.log_historical_returns, rowvar=False)
            target_return = self.log_target_return
        else:
            mean_returns = self.mean_returns
            cov_matrix = np.cov(self.historical_returns, rowvar=False)
            target_return = self.target_return


        # Цільова функція (мінімізація ризику)
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(weights.T @ (cov_matrix @ weights))

        # Умови
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Сума ваг = 1
            {'type': 'eq', 'fun': lambda w: (w @ mean_returns) - target_return}  # Дохідність = target_return
        ]

        # Обмеження на ваги
        bounds = [(0, 1) for _ in range(len(mean_returns))]

        # Початкове наближення
        initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

        # Оптимізація
        result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix),
                        method='SLSQP', bounds=bounds, constraints=constraints)

        return result