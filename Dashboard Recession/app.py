import pandas as pd
import numpy as np
import matplotlib as plt
import ffn
import scipy
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



from Investment import Backtestlib

import streamlit as st

st.title('Historical Recession Dashboard :')

st.write(":monkey_face: The'Historical Recession Dashboard' provides an interactive exploration of asset performance during past recessions, offering valuable insights into market behaviors during these downturns. With features that allow for the analysis of asset prices, investment returns, and key economic indicators, users can gain a comprehensive understanding of market dynamics in adverse conditions. While the dashboard offers detailed historical analysis, it's important to remember that past performance does not predict future market behaviors. The tool aims to equip users with knowledge to make more informed decisions and develop robust strategies for navigating potential future recessions")



Evens =  pd.read_excel("Data/recession.xlsm", index_col=0)


df_asset_1 = pd.read_csv("Data/EVEN_TEST_New_1.csv", index_col="Date")
df_asset_2 =  pd.read_csv("Data/EVEN_TEST_New_2.csv", index_col="Date")
df_asset_2["Price Close"] = df_asset_2["Close Price"]
df_asset = pd.concat([df_asset_1,  df_asset_2])[["Instrument", "Price Close"]]
df_asset.index = pd.to_datetime(df_asset.index)
df_asset.index = df_asset.index.date






tab1, tab2 = st.tabs(["Recession Indicator","EVENT ANALYSIS", "ASSET BEHAVIOR"])









with tab1:
    
    st.title('EVENT ANALYSIS :')
    st.write('Tab นี้เราสร้างมาเพื่อสังเกตพฤติกรรมของสินทรัพย์เมื่อเกิดเหตุถการ์ Recession ในอดีต:')
    
    
    option = st.selectbox(
        'Please select Even',
        (Evens.Event.tolist()))

    st.write('You selected:', option)
    
    
    
    show_btntab1 = st.button("Run!")
    lst_return = [] 
    
    if  show_btntab1: 
        def plot_asset_price(Evens, df_asset, option, i_start=360, i_end=360):

                Event = Evens[Evens["Event"] == option]
                dateEvent = Evens[Evens.Event == option].index[0]

                assets = df_asset.Instrument.unique()
                chunks = [assets[i:i + 3] for i in range(0, len(assets), 3)]

                for chunk in chunks:
                    cols = st.columns(len(chunk))

                    for i, asset in enumerate(chunk):
                        with cols[i]:
                            st.write(asset)
                            price = df_asset[df_asset['Instrument'] == asset]
                            price["return"] = np.log(price["Price Close"] / price["Price Close"].shift(1))
                            price = price[price.index > pd.to_datetime(dateEvent) - pd.DateOffset(days=i_start)]
                            price = price[price.index < pd.to_datetime(dateEvent) + pd.DateOffset(days=i_end)]
                            before_event = price[price.index < pd.to_datetime(dateEvent)]
                            after_event = price[price.index >= pd.to_datetime(dateEvent)] 

                            # Plotting
                            plt.figure(figsize=(12, 6))
                            plt.plot(before_event.index, before_event["Price Close"], label='Before Event')
                            plt.plot(after_event.index, after_event["Price Close"], label='After Event')
                            plt.title(f"Price for {asset}")
                            plt.xlabel("Date")
                            plt.ylabel("Price")
                            plt.legend()
                            st.pyplot(plt)
                            plt.clf()
                            
        st.title('Price :')

        plot_asset_price(Evens, df_asset, option, i_start=360, i_end=360)

  
                  

        def plot_asset_return(Evens, df_asset, option, i_start=360, i_end=360):

                Event = Evens[Evens["Event"] == option]
                dateEvent = Evens[Evens.Event == option].index[0]

                assets = df_asset.Instrument.unique()
                chunks = [assets[i:i + 3] for i in range(0, len(assets), 3)]

                for chunk in chunks:
                    cols = st.columns(len(chunk))

                    for i, asset in enumerate(chunk):
                        with cols[i]:
                            st.write(asset)
                            price = df_asset[df_asset['Instrument'] == asset]
                            price["return"] = np.log(price["Price Close"] / price["Price Close"].shift(1))
                            price = price[price.index > pd.to_datetime(dateEvent) - pd.DateOffset(days=i_start)]
                            price = price[price.index < pd.to_datetime(dateEvent) + pd.DateOffset(days=i_end)]
                            before_event = price[price.index < pd.to_datetime(dateEvent)]
                            after_event = price[price.index >= pd.to_datetime(dateEvent)] 

                            # Plotting
                            plt.figure(figsize=(12, 6))
                            plt.plot(before_event.index, before_event["return"], label='Before Event')
                            plt.plot(after_event.index, after_event["return"], label='After Event')
                            plt.title(f"return for {asset}")
                            plt.xlabel("Date")
                            plt.ylabel("return")
                            plt.legend()
                            st.pyplot(plt)
                            plt.clf()

        st.title('Return :')
        plot_asset_return(Evens, df_asset, option, i_start=360, i_end=360)                  

    
    
    
    
    
    
    
#############################   tab2   ASSET BEHAVIOR   #######################################################
    
with tab2:
    st.title(' ASSET BEHAVIOR :national_park:')
    st.write("เราจะศึกษาถึง worst case scenario ของ Asset class เมื่อเกิด Recession ในอดีตที่ผ่านมามีพฤติกรรมอย่างไรเพื่อให้เข้าใจ ผลตอบแทนและความเสี่ยงของสินทรัพย์แต่ละชนิด")
    

    option_asset = st.selectbox(
        'Please select ASSET',
        (df_asset.Instrument.unique().tolist()))

    st.write('You selected:', option_asset)
    
    price = df_asset[df_asset['Instrument'] == option_asset ]
    Price_return = np.log( price["Price Close"] /  price["Price Close"].shift(1))
    
    df_Drawdown = Backtestlib.drawdown(Price_return)
    topdrowdown , df_dd = Backtestlib.group_drawdown(Price_return)
    df_Drawdown["Drawdown"] = df_Drawdown["Drawdown"]*100
    
    

    # Create a Plotly figure
    figprice = px.line(price, x=price .index, y=price["Price Close"])
    # Display the figure in Streamlit
    st.plotly_chart(figprice)
    
    
    fig_Drawdown = px.area(df_Drawdown, x=df_Drawdown.index , y='Drawdown')
    fig_Drawdown.update_traces(line_color='red', fillcolor='rgba(255, 0, 0, 0.5)')  
    st.plotly_chart(fig_Drawdown)
    
    
    # Create a Plotly Table
    figtable_dd = go.Figure(data=[go.Table(
        header=dict(values=df_dd.columns.tolist(),
                    fill_color='white',
                    align='left'),
        cells=dict(values=[df_dd[col].tolist() for col in df_dd.columns],
                   fill_color='white',
                   align='left'))
                         ])
    st.plotly_chart(figtable_dd)
    
    
    show_btn = st.button("Simulation return!")
    
    if show_btn: 
        
        Evens['recession_start'] = (Evens['Event'] > Evens['Event'].shift(1))
        recession_starts = Evens[Evens['recession_start'] == True]
        recession_starts = recession_starts
        
        dic_aa = {} 
        def process_price_data(price, recession_starts, dic_aa):

            data_i = price

            for date in recession_starts.index:

                lst = []

                for i_start in range(5, 55, 1):
                    dic_of_return = {}

                    for i_End in range(5, 55, 1):
                        dic_of_return[i_End] = data_i[data_i.index > pd.to_datetime(date) - 
                                                      pd.DateOffset(days=i_start)].head(i_start+i_End)["Price Close"].pct_change(i_start+i_End -1)[-1]

                    df = pd.DataFrame(dic_of_return, index=["return"]).T
                    df["day after event"] = df.index
                    df["day before event"] = i_start

                    lst.append(df)

                aa = pd.concat(lst).pivot_table(values='return', index=['day after event'], columns='day before event') *100
                Event = recession_starts[recession_starts.index == date].Event[0]
                print(Event)
                dic_aa[Event] = aa

                mask = np.triu(np.ones_like(aa, dtype=bool))

            return dic_aa
        
        dic_aa = process_price_data(price, recession_starts, dic_aa)
        
        st.header("Run finish" )
        

                


        def show_heatmap(dic_aa):
            # Create a list of the dates/dataframes (or numpy arrays) in dic_aa
            data = list(dic_aa.items())

            # Split the list into chunks of 3 or less
            chunks = [data[i:i + 3] for i in range(0, len(data), 3)]

            for chunk in chunks:
                # Create a row with 3 columns
                cols = st.columns(len(chunk))

                for i, (date, aa) in enumerate(chunk):
                    # Set up the matplotlib figure
                    with cols[i]:
                        f, ax = plt.subplots(figsize=(11, 9))
                        plt.title(date)
                        sns.set_style('whitegrid')

                        # Generate a custom diverging colormap
                        cmap = sns.diverging_palette(230, 20, as_cmap=True)

                        # Define mask
                        mask = np.triu(np.ones_like(aa, dtype=bool))

                        # Draw the heatmap with the mask and correct aspect ratio
                        sns.heatmap(aa, mask=mask, cmap="PiYG", vmax=.3, center=0,
                                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

                        # Display the figure with Streamlit
                        st.pyplot(f)




        # Assuming that dic_aa is a dictionary with date keys and dataframes (or numpy arrays) as values.
        # Call the function to display the heatmaps in your Streamlit app
        show_heatmap(dic_aa)






