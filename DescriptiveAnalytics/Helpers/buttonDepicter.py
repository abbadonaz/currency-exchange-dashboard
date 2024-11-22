import streamlit as st


def display_buttons(total_days, avg_ask, avg_bid):
    """
        Displays metrics for total days, average ask price, and average bid price in a three-column layout.

        Parameters:
        - total_days (int): Total number of days in the data range.
        - avg_ask (float): Average 'Ask' price in PLN.
        - avg_bid (float): Average 'Bid' price in PLN.
        """
    total_col, ask_col, bid_col = st.columns(3, gap="large")

    with total_col:
        st.info('Days')
        st.metric(label='Total Days', value=total_days)

    with ask_col:
        st.info("'Ask' Price")
        st.metric(label='Avg. Ask Price (PLN)', value=f"{avg_ask:,.3f}")

    with bid_col:
        st.info("'Bid' Price")
        st.metric(label='Avg. Bid Price (PLN)', value=f"{avg_bid:,.3f}")
