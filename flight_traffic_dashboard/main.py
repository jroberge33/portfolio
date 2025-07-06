'''
Flight Traffic Dashboard Main

'''

# import libraries

from flightapi import FLIGHTAPI
import pandas as pd
import sankey as sk
import seaborn as sns
import matplotlib.pyplot as plt
import panel as pn
import hvplot.pandas
import holoviews as hv
import geoviews as gv
import geopandas as gpd

# list of busiest states
BUSIEST_STATES = ["GA", "CA", "TX", "CO", "IL", "NY", "FL", "NC"]

pn.extension()

# create flight API object
flight = FLIGHTAPI()

# file paths for flight data
flight_paths = ['hw3_data/12_2018.csv', 'hw3_data/06_2019.csv',
                'hw3_data/12_2019.csv', 'hw3_data/06_2020.csv',
                'hw3_data/12_2020.csv', 'hw3_data/06_2021.csv',
                'hw3_data/12_2021.csv', 'hw3_data/06_2022.csv',
                'hw3_data/12_2022.csv', 'hw3_data/06_2023.csv',
                'hw3_data/12_2023.csv', 'hw3_data/06_2024.csv']

# load object and flight_df
flight.load_flights(flight_paths)
flight_df = flight.get_data().reset_index()

# Panel Widgets for Sankey Diagram

state_widget = pn.widgets.MultiChoice(
    name="Select Up to 10 States",
    options=sorted(set(flight_df["ORIGIN_STATE"]).union(set(flight_df["DEST_STATE"]))),
    value=BUSIEST_STATES,
    max_items=10
)

month_widget = pn.widgets.MultiChoice(
    name="Select Month(s)",
    options=sorted(flight_df["MONTH"].unique(), reverse=True),
    value=[flight_df["MONTH"].max()]
)

middle_node_selector = pn.widgets.Select(
    name="Segment Flow By",
    options=["None", "MONTH", "DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "DISTANCE_GROUP"],
    value="None"
)

cancellation_filter = pn.widgets.RadioButtonGroup(
    name="Flight Cancellation Filter",
    options=["All Flights", "Non-Cancelled Only", "Cancelled Only"],
    value="All Flights",
    button_type="light",
    orientation="vertical"
)

# Callback function for sankey diagram
def get_flight_data(state, month, cancellation, middle):

    # load filtered flight data by month, cancellation type, states
    filtered_data, cols = flight.filter_flights(state, month, cancellation, middle)

    # produce sankey fig with make_sankey function
    fig = sk.make_sankey(filtered_data, *cols, vals="flights")
    return fig

# bind sankey callback function to widgets
sankey_plot = pn.bind(get_flight_data, state_widget, month_widget, cancellation_filter, middle_node_selector)

# create widgets for geomap tab
delay_type = pn.widgets.Select(name='Delay Type', options=flight.get_delay_types(), value = 'WEATHER_DELAY')
avg_total_delay = pn.widgets.Select(name="Average or Total Delays", options = ['TOTAL_DELAY', 'AVG_DELAY'], value = 'TOTAL_DELAY')
delay_length = pn.widgets.IntSlider(name = 'Time Delayed', start = 0, end = 500, step = 10, value = 10)

# callback function for geomap
def create_map(delay_type, avg_total_delay, delay_length):

    # create filtered df by delay_type and length
    filtered_df = flight.group_monthly_delays(delay_type, avg_total_delay, delay_length)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df['LONGITUDE'], filtered_df['LATITUDE']))

    # Create Holoviews scatter plot
    plot = gdf.hvplot.points(
        x='LONGITUDE', y='LATITUDE',
        geo=True,
        tiles='CartoLight',
        color=avg_total_delay, # color by delay so larger delays are darker
        hover_cols=['ORIGIN_CITY_NAME', avg_total_delay] # columns the user can hover over to get more info
    )

    return plot

# bind geomap callback function to widgets
map = pn.bind(create_map, delay_type, avg_total_delay, delay_length)

# Create boxplot tab widgets
airline_widget = pn.widgets.MultiChoice(
    name="Select Airline(s)",
    options=sorted(flight_df["OP_UNIQUE_CARRIER"].unique()),
    value=list(flight_df["OP_UNIQUE_CARRIER"].unique())
)

day_widget = pn.widgets.MultiChoice(
    name="Select Day(s) of the Week",
    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    value=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

no_max_state_widget = pn.widgets.MultiChoice(
    name="Select States",
    options=sorted(set(flight_df["ORIGIN_STATE"]).union(set(flight_df["DEST_STATE"]))),
    value=BUSIEST_STATES,
)

outlier_widget = pn.widgets.Checkbox(name="Show Outliers", value=False)

# Create boxplot callback function
def generate_boxplots(selected_states, selected_months, selected_airlines, selected_days, show_outliers):

    # create filtered df by selected states, months, airlines and days of week
    filtered_df = flight.get_boxplot_data(selected_states, selected_months, selected_airlines, selected_days, show_outliers)

    # produce one boxplot for departures
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    sns.boxplot(x="CAUSE_OF_DELAY", y="DEP_DELAY_NEW", data=filtered_df, ax=axes[0], whis=[5,95], showfliers=show_outliers)
    axes[0].set_title("Departure Delay by Delay Type", fontsize=10)
    axes[0].set_xlabel("Delay Type", fontsize=8)
    axes[0].set_ylabel("Departure Delay (minutes)", fontsize=8)
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0].tick_params(axis='y', labelsize=8)

    # produce a second adjacent boxplot for arrivals
    sns.boxplot(x="CAUSE_OF_DELAY", y="ARR_DELAY_NEW", data=filtered_df, ax=axes[1], whis=[5,95], showfliers=show_outliers)
    axes[1].set_title("Arrival Delay by Delay Type", fontsize=10)
    axes[1].set_xlabel("Delay Type", fontsize=8)
    axes[1].set_ylabel("Arrival Delay (minutes)", fontsize=8)
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    return fig

# bind together boxplot callback function and widgets
boxplot = pn.bind(generate_boxplots, no_max_state_widget, month_widget, airline_widget, day_widget, outlier_widget)

card_width = 320

# produce cards to go on the sidebar
sankey_card = pn.Card(
    pn.Column(
        state_widget,
        month_widget,
        cancellation_filter,
        middle_node_selector
    ),
    title="Sankey", width=card_width, collapsed=False
)
plot_card = pn.Card(
    pn.Column(
        delay_type,
        avg_total_delay,
        delay_length
    ),

    title="Geomap", width=card_width, collapsed=True
)

boxplot_card = pn.Card(
    pn.Column(
        no_max_state_widget,
        month_widget,
        airline_widget,
        day_widget,
        outlier_widget
    ),

    title="Boxplot", width=card_width, collapsed=True
)

# dashboard layout with each tab and associated sidebar
layout = pn.template.FastListTemplate(
    title="Flight Explorer",
    sidebar=[
        sankey_card,
        plot_card,
        boxplot_card
    ],
    theme_toggle=False,
    main=[
        pn.Tabs(
            ("Sankey", sankey_plot),
            ("Map", map),
            ("Boxplot", boxplot),
            active=1
        )

    ],
    header_background='#a93226'

).servable()

layout.show()
