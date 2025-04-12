import numpy as np
from scipy.optimize import minimize
import sys
from scipy.stats import shapiro, norm
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, historical_returns, target_return, alpha):
        self.historical_returns = historical_returns
        self.target_return = target_return
        self.alpha = alpha

        self.mean_returns = np.mean(historical_returns, axis=0)  # очікувані дохідності
        
        if all(target_return > element for element in self.mean_returns):
            print("ERROR")
            sys.exit()


    def plot_efficient_frontier(self, num_portfolios=100, optimize_func=None):
        if optimize_func is None:
            optimize_func = self.optimize_portfolio_by_Markowitz_1

        _, _, log_mean_returns = self.logarithmization()
        target_returns = np.linspace(log_mean_returns.min(), log_mean_returns.max(), num_portfolios)
        efficient_portfolios = []

        for target_return in target_returns:
            result = optimize_func(_target_return=target_return)
            if result is not None:
                efficient_portfolios.append((result['fun'], target_return))

        risks, returns = zip(*efficient_portfolios)

        plt.figure(figsize=(10, 6))
        plt.plot(risks, returns, 'o-', markersize=5, label='Efficient Frontier')
        plt.title('Efficient Frontier')
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.show()



    def normal_distribution(self):# перевірка нормального розподілу
        _, p_with_lock = shapiro(self.historical_returns)
        print("Normality with locking:", "Passed" if p_with_lock > 0.05 else "Failed")
        return p_with_lock > 0.05

    def calculate_volatility(self, result):
        """
        Calculate the volatility (standard deviation) of a portfolio's returns.
        This method computes the portfolio's volatility based on historical returns
        and the weights of the assets in the portfolio.
        Args:
            result (numpy.ndarray): A 1D array representing the weights of each asset 
                                    in the portfolio. The weights should sum to 1.
        Returns:
            float: The standard deviation of the portfolio's returns, representing 
                   its volatility.
        """
        # вычисляем фактическое изменение в цене каждой акции в портфеле (в процентах), после чего умножаем на минус (чем больше число - тем больеше падение)
        historical_returns_array = self.historical_returns.values
        # вычисляем фактическое изменение в цене портфеля (в единицах - деньги), то бишь умножаем изменение акций на их вес в портфеле и суммируем по всем акциям
        RP = (historical_returns_array * result).sum(axis=1) 

        # вычисляем стандартное отклонение (волатильность) портфеля
        return RP.std()

    def calculate_VaR(self, result):
        z_score = norm.ppf(1 - self.alpha)

        # вычисляем фактическое изменение в цене каждой акции в портфеле (в процентах), после чего умножаем на минус (чем больше число - тем больеше падение)
        historical_returns_array = -self.historical_returns.values
        # вычисляем фактическое изменение в цене портфеля (в единицах - деньги), то бишь умножаем изменение акций на их вес в портфеле и суммируем по всем акциям
        RP = (historical_returns_array * result['x']).sum(axis=1) 

        if self.normal_distribution():
            # Parametric method (normal distribution)
            var = z_score * result['fun'] * np.sqrt(RP.shape[0] + 1)
            method = "Parametric method (normal distribution)"
        else:
            # Historical method (distribution-independent)
            RP_d = np.sort(RP)[::-1]
            var = np.quantile(RP_d, (1 - self.alpha))
            method = "Historical method (distribution-independent)"

        return var, method

    
    def logarithmization(self, _mean_returns=None, _cov_matrix=None, _target_return=None):
        log_returns = np.log(self.historical_returns + 1)
        log_target_return = np.log(self.target_return + 1)
        log_mean_returns = np.log(self.mean_returns + 1)

        if _mean_returns is not None:
            log_mean_returns = np.log(_mean_returns + 1)            
        if _target_return is not None:
            log_target_return = np.log(_target_return + 1)
        if _cov_matrix is not None:
            log_returns = np.log(_cov_matrix + 1)

        return log_returns, log_target_return, log_mean_returns

    def optimize_portfolio_by_Markowitz_1(self, _mean_returns=None, _cov_matrix=None, _target_return=None, log=True):
        # Логарифмізація
        if log:
            log_historical_returns, target_return, mean_returns = self.logarithmization(_mean_returns, _cov_matrix, _target_return)
            cov_matrix = np.cov(log_historical_returns, rowvar=False)
        else:
            mean_returns = _mean_returns or self.mean_returns
            cov_matrix = _cov_matrix or np.cov(self.historical_returns, rowvar=False)
            target_return = _target_return or self.target_return

        
        # Testing ds
        # mean_returns = np.array([0.15, 0.1, 0.12])
        # cov_matrix = np.array([[0.05, 0.01, 0.02], [0.01, 0.04, 0.015], [0.02, 0.015, 0.03]])
        # target_return = 0.11
        # T=5


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

        return {'x': result.x,
                'fun': result.fun}
    
    def optimize_portfolio_by_Markowitz_3(self, _mean_returns=None, _cov_matrix=None, _target_return=None, log=True, T=50):
        # Логарифмізація
        if log:
            log_historical_returns, _, mean_returns = self.logarithmization(_mean_returns, _cov_matrix, _target_return)
            cov_matrix = np.cov(log_historical_returns, rowvar=False)
        else:
            mean_returns = _mean_returns or self.mean_returns
            cov_matrix = _cov_matrix or np.cov(self.historical_returns, rowvar=False)

        # Testing ds
        # mean_returns = np.array([0.15, 0.1, 0.12])
        # cov_matrix = np.array([[0.05, 0.01, 0.02], [0.01, 0.04, 0.015], [0.02, 0.015, 0.03]])
        # target_return = 0.11
        # T=5

        cov_matrix_inv = np.linalg.solve(cov_matrix, np.eye(cov_matrix.shape[0]))
        ones = np.ones_like(mean_returns)
        divider = ones.T @ cov_matrix_inv @ ones

        # Перший доданок
        first_term = (cov_matrix_inv @ ones) / divider

        # Другий доданок
        R = cov_matrix_inv - (np.outer(cov_matrix_inv @ ones, ones) @ cov_matrix_inv) / divider
        second_term =(R @ mean_returns) / (2*T)

        result = first_term + second_term
        return {'x': result,
                'fun': self.calculate_volatility(result)}
        
    def optimize_portfolio_by_VaR_min(self, _mean_returns=None, _cov_matrix=None, _target_return=None, log=True):
        # Логарифмізація
        if log:
            log_historical_returns, _, mean_returns = self.logarithmization(_mean_returns, _cov_matrix, _target_return)
            cov_matrix = np.cov(log_historical_returns, rowvar=False)
        else:
            mean_returns = _mean_returns or self.mean_returns
            cov_matrix = _cov_matrix or np.cov(self.historical_returns, rowvar=False)

        
        # Testing ds
        # mean_returns = np.array([0.15, 0.1, 0.12])
        # cov_matrix = np.array([[0.05, 0.01, 0.02], [0.01, 0.04, 0.015], [0.02, 0.015, 0.03]])
        # target_return = 0.11
        # T=5

        cov_matrix_inv = np.linalg.solve(cov_matrix, np.eye(cov_matrix.shape[0]))
        ones = np.ones_like(mean_returns)
        divider = ones.T @ cov_matrix_inv @ ones

        # Перший доданок
        first_term = (cov_matrix_inv @ ones) / divider

        # Другий доданок
        R = cov_matrix_inv - (np.outer(cov_matrix_inv @ ones, ones) @ cov_matrix_inv) / divider
        z = norm.ppf(1 - self.alpha)
        coefficient = np.sqrt(1 / divider) / np.sqrt(z**2 - mean_returns.T @ R @ mean_returns)
        second_term = coefficient * R @ mean_returns

        result = first_term + second_term


        return {'x': result,
                'fun': self.calculate_volatility(result)}