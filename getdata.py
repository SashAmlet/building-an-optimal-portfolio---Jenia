import yfinance as yf
import pandas as pd
import numpy as np

class DataLoader:
    @staticmethod
    def get_historical_returns(tickers, start_date, end_date):
        # Завантаження даних про ціни акцій
        data = yf.download(tickers, start=start_date, end=end_date)['Close']

        # Перевірка, чи дані завантажилися коректно
        if data.empty:
            print("Дані не завантажилися. Перевірте тикери та дати.")
            return np.array([])

        # Обчислення щоденних дохідностей (у відсотках) на основі зміни ціни на акції
        clean_data = data.dropna()
        returns = (clean_data / clean_data.shift(1) - 1).iloc[1:] #щоб почати з 1 рядку (рахунок з 0)

        return returns

    @staticmethod
    def export_to_excel(historical_returns, tickers, filename='historical_returns.xlsx'):
        # Перетворення масиву в DataFrame
        df = pd.DataFrame(historical_returns, columns=tickers)

        # Запис DataFrame в Excel
        df.to_excel(filename, index=False)

        print(f"Дані успішно експортовані в файл {filename}")
