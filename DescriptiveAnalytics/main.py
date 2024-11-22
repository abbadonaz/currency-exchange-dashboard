import os
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
from Helpers.watchlist import CurrencyDataDownloader
from Helpers.dataLoader import load_data
from Helpers.statisticsComputer import StatisticsComputer
from Helpers.dataPlotter import display_graph
from Helpers.buttonDepicter import display_buttons

st.set_page_config(page_title='Exchange Rate Dashboard', layout="wide")
st.sidebar.image("Helpers/images/logo1.png", caption='Data Analytics')
st.write("# Currency Exchange Rate Dashboard")
st.write('### The exchange rate is calculated with respect to Polish Zloty (PLN)')
option = st.selectbox("Please select the foreign currency:", ("USD", "GBP", "EUR", "JPY","TRY", "CHF"))

# Date input for selecting the range
st.write("Please select the start and end dates for the currency data:")
start_date = st.date_input("Start Date", min_value=pd.to_datetime('1990-01-01'))
end_date = st.date_input("End Date", min_value=start_date)

# Checking if the dates are valid and making sure start_date is before end_date
if start_date > end_date:
    st.error("Start date cannot be later than the end date. Please select a valid date range.")
else:
    # Show feedback to the user while processing data
    st.write(f"Fetching data for the period: {start_date} to {end_date}")

    # Trigger the currency data download (using the user-selected date range)
    currencyDownloader = CurrencyDataDownloader(startDate=start_date, endDate=end_date, currency_querried=option)

    # Check if the data files already exist
    output_path = f'C:/Users/User/currencyWatchlist/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download the currency data if not already available
    currencyDownloader.download_all_currencies(output_path=output_path)
    st.success("Data download completed!")

    # Load data after downloading or if files already exist
    df_main = load_data(os.path.join(output_path, f'exchange_rate_data_{currencyDownloader.currency_querried}.csv'))
    st.write('Depict Statistics')

    # Create columns for side-by-side DataFrames and display them with highlights, stats
    stats_main = StatisticsComputer(df_main)

    # Access statistics
    total_main = stats_main.total_days_value
    ask_main = stats_main.ask_mean_value
    bid_main = stats_main.bid_mean_value

    display_buttons(total_main,ask_main,bid_main)

    st.markdown(f"### {currencyDownloader.currency_querried} Exchange Rate")
    st.text('Exchange rates provided by the National Bank of Poland')
    st.dataframe(df_main.style.highlight_max(axis=0, subset=['Ask', 'Bid'], color='green'))

    display_graph(df_main=df_main, title=currencyDownloader.currency_querried)

    # Display a brief description
    st.markdown("""
        **Currency Exchange Rates** are shown with the highest values in 'Ask' and 'Bid' columns highlighted in green.
        Use this dashboard to explore exchange rate fluctuations between selected dates.
        """)
