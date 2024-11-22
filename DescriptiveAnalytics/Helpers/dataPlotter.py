import pandas as pd
import plotly.express as px
import streamlit as st


def display_graph(df_main, title):
    if 'Date' in df_main.columns:
        df_main['Date'] = pd.to_datetime((df_main['Date']))
        fig = px.line(df_main,
                      x='Date',
                      y=['Ask', 'Bid'],
                      labels={'value': 'Exchange Rate (PLN)', 'Date': 'Date'},
                      title=f"{title} Exchange Rate Over Time",
                      template="plotly_white"
                      )
        fig.update_layout(legend_title_text='Rate Type', legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("The dataset does not contain a 'Date' column. Unable to plot")
