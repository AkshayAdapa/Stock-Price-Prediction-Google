# Stock Price Prediction - Google

## Overview
This project aims to predict Google stock prices using time series analysis and machine learning techniques. The dataset includes various details such as opening price, closing price, and trading volume of Google stocks.

## Dataset
The dataset used for this project was obtained from Yahoo Finance using the `yfinance` library. It spans from May 26, 2014, to May 23, 2024, covering a wide range of historical data points.

## Project Structure
- **Notebook**: `stock.ipynb`
  - This Jupyter Notebook contains the entire workflow for data fetching, preprocessing, model building, training, evaluation, and saving.
  
- **Libraries Used**:
  - `numpy` for numerical operations
  - `pandas` for data manipulation and analysis
  - `matplotlib` for data visualization
  - `yfinance` for fetching stock data
  - `sklearn` for preprocessing and metrics
  - `keras` for building and training LSTM model

## Steps Involved
1. **Data Fetching and Preprocessing**:
   - Fetching historical stock data using `yfinance`.
   - Resetting index and selecting relevant features (Close price and Volume).

2. **Visualization**:
   - Visualizing moving averages (50-day, 100-day, 200-day) compared with closing prices to understand trends.

3. **Normalization**:
   - Normalizing the data using `MinMaxScaler` to prepare it for LSTM model training.

4. **Model Building**:
   - Building an LSTM model with multiple layers and dropout for regularization.

5. **Model Training**:
   - Training the LSTM model on the prepared data with `adam` optimizer and `mean_squared_error` loss function.

6. **Model Evaluation**:
   - Evaluating the model performance using RMSE (Root Mean Square Error) on both training and test sets.

7. **Model Saving**:
   - Saving the trained model in the native Keras format (`Stock_Prediction_Model.keras`).

## Results
- **Train RMSE**: 3.06
- **Test RMSE**: 5.82

These results indicate that the model performs reasonably well in predicting Google stock prices based on historical data.

## Future Improvements
- Explore additional features or alternative models (such as GRU or Transformer-based models) to further improve prediction accuracy.
- Fine-tune hyperparameters or experiment with different architectures to optimize the model.
- Incorporate external factors (news sentiment, economic indicators) for more robust predictions.

## References
- [yfinance documentation](https://pypi.org/project/yfinance/)
- [Keras API documentation](https://keras.io/api/)
- [Scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)

## Contributors
- Akshay Kumar

Feel free to contribute, suggest improvements, or use this project for learning and experimentation with time series analysis and machine learning.
