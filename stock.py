import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Set background image with transparency
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url('https://c8.alamy.com/comp/2BF0R22/investing-balance-as-a-business-see-saw-and-economic-stock-market-or-bull-and-bear-economy-on-a-see-saw-concept-with-3d-illustration-elements-2BF0R22.jpg');
#         background-size: cover;
#         color: #000000; /* Thick black text */
#     }
#     .title {
#         color: #ff0000; /* Red title */
#         font-weight: bold; /* Bold title */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Centered image and title
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://cdn.pixabay.com/photo/2024/01/06/02/44/ai-generated-8490532_640.png" alt="Stock Market Predictor" width="300">
        <h1 style="font-size: 36px; color: #ff0000; font-weight: bold;">Stock Market Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

stock = st.text_input('**Enter Stock Symbol**', 'GOOG')
start = '2014-05-08'
end = '2024-05-23'  # Requested end date

try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error("No data found for the specified date range. Please try another range or stock symbol.")
    else:
        st.subheader('**Stock Data**')
        st.write(data)

        # Check if the latest available date matches the requested end date
        latest_date = data.index[-1]
        if latest_date < pd.to_datetime(end):
            st.warning(f"Data is available only up to {latest_date.date()}, not the requested end date {end}.")

        data_train = data.Close.values[:int(len(data)*0.80)].reshape(-1, 1)
        data_test = data.Close.values[int(len(data)*0.80):].reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train_scaled = scaler.fit_transform(data_train)
        data_test_scaled = scaler.transform(data_test)

        # Button selection for graphs
        selected_graph = st.radio(
            "**Select Graph**",
            ("Price vs MA50", "Price vs MA50 vs MA100", "Price vs MA100 vs MA200", "Original Price vs Predicted Price")
        )

        if selected_graph == "Price vs MA50":
            st.subheader('**Price vs MA50**')
            st.write("""
            This graph shows the stock price (in green) along with its 50-day moving average (MA50) calculated as the average of the closing prices over the last 50 days. 
            """)
            ma_50_days = data.Close.rolling(50).mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ma_50_days, 'r', label='MA50')
            ax.plot(data.Close, 'g', label='Price')
            ax.set_xlabel('**Date**')
            ax.set_ylabel('**Price**')
            ax.set_title('**Price vs MA50**')
            ax.legend()
            st.pyplot(fig)

            # Show relevant statistics
            st.subheader('**Relevant Statistics**')
            st.write('**Mean Price:**', data.Close.mean())
            st.write('**Standard Deviation:**', data.Close.std())

        elif selected_graph == "Price vs MA50 vs MA100":
            st.subheader('**Price vs MA50 vs MA100**')
            st.write("""
            This graph shows the stock price (in green) along with its 50-day moving average (MA50, in red) and 100-day moving average (MA100, in blue). 
            """)
            ma_50_days = data.Close.rolling(50).mean()
            ma_100_days = data.Close.rolling(100).mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ma_50_days, 'r', label='MA50')
            ax.plot(ma_100_days, 'b', label='MA100')
            ax.plot(data.Close, 'g', label='Price')
            ax.set_xlabel('**Date**')
            ax.set_ylabel('**Price**')
            ax.set_title('**Price vs MA50 vs MA100**')
            ax.legend()
            st.pyplot(fig)

            # Show relevant statistics
            st.subheader('**Relevant Statistics**')
            st.write('**Mean Price:**', data.Close.mean())
            st.write('**Standard Deviation:**', data.Close.std())

        elif selected_graph == "Price vs MA100 vs MA200":
            st.subheader('**Price vs MA100 vs MA200**')
            st.write("""
            This graph shows the stock price (in green) along with its 100-day moving average (MA100, in red) and 200-day moving average (MA200, in blue). 
            """)
            ma_100_days = data.Close.rolling(100).mean()
            ma_200_days = data.Close.rolling(200).mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ma_100_days, 'r', label='MA100')
            ax.plot(ma_200_days, 'b', label='MA200')
            ax.plot(data.Close, 'g', label='Price')
            ax.set_xlabel('**Date**')
            ax.set_ylabel('**Price**')
            ax.set_title('**Price vs MA100 vs MA200**')
            ax.legend()
            st.pyplot(fig)

            # Show relevant statistics
            st.subheader('**Relevant Statistics**')
            st.write('**Mean Price:**', data.Close.mean())
            st.write('**Standard Deviation:**', data.Close.std())

        else:
            st.subheader('**Original Price vs Predicted Price**')
            st.write("""
            This graph shows the original stock price (in green) and the predicted stock price (in yellow) based on a trained machine learning model.
            """)
            # Prepare data for prediction
            x_test = []
            y_test = []

            for i in range(100, len(data_test_scaled)):
                x_test.append(data_test_scaled[i-100:i, 0])
                y_test.append(data_test_scaled[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Load the Keras model
            model = load_model("Stock_Prediction_Model.keras")

            # Make predictions
            predictions = model.predict(x_test)
            predictions = np.reshape(predictions, (-1, 1))  # Reshape predictions to a 2D array with one column
            predictions = scaler.inverse_transform(predictions)
            y_test = np.reshape(y_test, (-1, 1))  # Reshape y_test to a 2D array with one column
            y_test = scaler.inverse_transform(y_test)

            df = pd.DataFrame({'Date': data.index[100+len(data)-len(data_test_scaled):], 'Original Price': y_test.reshape(-1), 'Predicted Price': predictions.reshape(-1)})

            source = df.melt('Date', var_name='Price Type', value_name='Price')

            brush = alt.selection(type='interval', encodings=['x'])

            base = alt.Chart(source).mark_line().encode(
                x='Date:T',
                y='Price:Q',
                color=alt.Color('Price Type:N', scale=alt.Scale(domain=['Original Price', 'Predicted Price'], range=['#3CAEA3', '#F6D55C']))
            ).properties(
                width=800,
                height=400
            )

            points = base.mark_point().encode(
                opacity=alt.condition(brush, alt.value(1), alt.value(0))
            ).add_selection(
                brush
            )

            st.altair_chart(points, use_container_width=True)

            # Show relevant statistics
            st.subheader('**Relevant Statistics**')
            st.write('**Mean Original Price:**', np.mean(y_test))
            st.write('**Mean Predicted Price:**', np.mean(predictions))
            st.write('**Standard Deviation (Original Price):**', np.std(y_test))
            st.write('**Standard Deviation (Predicted Price):**', np.std(predictions))

except Exception as e:
    st.error("**An error occurred:** {}".format(e))
