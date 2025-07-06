"""
Flight data API Program
"""

# import libraries
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import pickle
import os

# day of week mapping dct
DAY_OF_WEEK_MAPPING = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday"
}

class FLIGHTAPI:
    def __init__(self):
        self.flight = None

    def load_flights(self, file_paths):
        """Load multiple flight datasets from CSV files and concatenate them into one df."""

        # due to large size of file, pkl used to more efficiently load data
        pkl_file_path = 'hw3_data/flight_df.pkl'

        # Load existing flight_df if the file exists
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                flight_df = pickle.load(f)
            self.flight = flight_df

        # if the file doesn't exist, load and clean the data again
        else:
            dfs = [pd.read_csv(file) for file in file_paths]
            flight_df = pd.concat(dfs, ignore_index=True)
            self.flight = flight_df
            self.clean_data()
            with open(pkl_file_path, 'wb') as f:
                pickle.dump(self.flight, f)

    def clean_data(self):
        """Perform data cleaning and transformation."""
        df = self.flight

        # convert data types
        df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype("category")
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df["MONTH"] = df["FL_DATE"].dt.strftime('%Y-%m')  # Extract Year-Month

        df["OP_UNIQUE_CARRIER"] = df["OP_UNIQUE_CARRIER"].astype("category")
        df["TAIL_NUM"] = df["TAIL_NUM"].astype("string")

        df["ORIGIN"] = df["ORIGIN"].astype("category")
        df["ORIGIN_CITY_NAME"] = df["ORIGIN_CITY_NAME"].astype("category")

        df["DEST"] = df["DEST"].astype("category")
        df["DEST_CITY_NAME"] = df["DEST_CITY_NAME"].astype("category")

        df["DEP_DELAY_NEW"] = df["DEP_DELAY_NEW"].astype("float64")
        df["ARR_DELAY_NEW"] = df["ARR_DELAY_NEW"].astype("float64")
        df["AIR_TIME"] = df["AIR_TIME"].astype("float64")
        df["DISTANCE"] = df["DISTANCE"].astype("int64")

        df["DISTANCE_GROUP"] = df["DISTANCE_GROUP"].astype("category")
        df["CANCELLATION_CODE"] = df["CANCELLATION_CODE"].astype("category")

        # create state specific variable
        df["ORIGIN_STATE"] = df["ORIGIN_CITY_NAME"].str.split(",").str[-1].str.strip()
        df["DEST_STATE"] = df["DEST_CITY_NAME"].str.split(",").str[-1].str.strip()

        # convert time to hh:mm
        df["DEP_TIME"] = df["DEP_TIME"].apply(
            lambda x: f"{int(x // 100):02}:{int(x % 100):02}" if pd.notna(x) else "00:00")
        df["ARR_TIME"] = df["ARR_TIME"].apply(
            lambda x: f"{int(x // 100):02}:{int(x % 100):02}" if pd.notna(x) else "00:00")

        # fill numeric missing with 0
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(0)

        # fill categorical missing with n/a
        cat_cols = df.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            df[col] = df[col].cat.add_categories("N/A").fillna("N/A")

        delay_columns = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
        # create cause of delay column that is the largest value (mins) of the other delay columns
        df["CAUSE_OF_DELAY"] = df[delay_columns].idxmax(axis=1)
        # Replace rows where all delays are zero with "None"
        df["CAUSE_OF_DELAY"] = np.where(df[delay_columns].max(axis=1) == 0, "None", df["CAUSE_OF_DELAY"])

        # code to find coordinate of lists, using goelocator and Nominatim library/API
        cities = list(set(list(df["ORIGIN_CITY_NAME"])))

        geolocator = Nominatim(user_agent="jroberge")
        pkl_path = 'hw3_data/city_coordinates_dct.pkl'

        # Load existing city coordinates if the file exists
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                city_coordinates = pickle.load(f)
        else:
            city_coordinates = {}

        # Geocode only cities that are not in the existing dictionary
        for city in cities:
            if city not in city_coordinates:
                location = geolocator.geocode(city, exactly_one=True, timeout=2, addressdetails=False)

                if location:
                    print(city)
                    city_coordinates[city] = (location.latitude, location.longitude)
                else:
                    city_coordinates[city] = (None, None)
                    print(f"Location not found for: {city}")

        # Save the updated city coordinates dictionary
        with open(pkl_path, 'wb') as f:
            pickle.dump(city_coordinates, f)

        # apply the dictionary to create new columns for latitude and longitude
        df['LATITUDE'] = df['ORIGIN_CITY_NAME'].apply(lambda x: city_coordinates.get(x, (None, None))[0])
        df['LONGITUDE'] = df['ORIGIN_CITY_NAME'].apply(lambda x: city_coordinates.get(x, (None, None))[1])

        # website provided cancellation and carrier codes to transform respective columns into understandable values
        cancellation_df = pd.read_csv('hw3_data/cancellation_codes.csv')
        carrier_df = pd.read_csv('hw3_data/carrier_codes.csv')
        cancellation_dict = dict(zip(cancellation_df['Code'], cancellation_df['Description']))
        carrier_dict = dict(zip(carrier_df['Code'], carrier_df['Description']))
        df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].map(cancellation_dict)
        df['OP_UNIQUE_CARRIER'] = df['OP_UNIQUE_CARRIER'].map(carrier_dict)
        df.set_index(["FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM"], inplace=True)

        self.flight = df

    def get_data(self):
        """Return the cleaned flight data."""
        return self.flight

    def filter_flights(self, selected_states, selected_months, cancellation_status, middle_node):
        """ Function to filter flights in sankey tab """
        df = self.flight.copy()
        filtered = df[
            (df["ORIGIN_STATE"].isin(selected_states)) &
            (df["DEST_STATE"].isin(selected_states)) &
            (df["MONTH"].isin(selected_months))
            ]

        # Define grouping columns
        if middle_node != 'None':
            group_cols = ["ORIGIN_STATE", middle_node, "DEST_STATE"]
        else:
            group_cols = ["ORIGIN_STATE", "DEST_STATE"]

        # Establish toggle for cancelled flight option
        if cancellation_status == "Non-Cancelled Only":
            filtered = filtered[filtered["CANCELLED"] == 0]
        elif cancellation_status == "Cancelled Only":
            filtered = filtered[filtered["CANCELLED"] == 1]

        # Aggregate flight counts
        filtered_data = filtered.groupby(group_cols).size().reset_index(name="flights")
        return filtered_data, group_cols


    def group_monthly_delays(self, delay_type, avg_total_delay, delay_length):
        """ Function to group monthly delays with long, lat in geomap tab """

        # group by city and cause of delay, summing delay mins, geting an avg, and keeping long, lat values
        grouped_df = self.flight.groupby(['ORIGIN_CITY_NAME', 'CAUSE_OF_DELAY']).agg(
            TOTAL_DELAY=('DEP_DELAY_NEW', 'sum'),
            AVG_DELAY=('DEP_DELAY_NEW', 'mean'),
            LATITUDE=('LATITUDE', 'first'),
            LONGITUDE=('LONGITUDE', 'first')
        ).reset_index()

        # filter to specific delay type and length
        grouped_df = grouped_df[(grouped_df['CAUSE_OF_DELAY'] == delay_type) & (grouped_df[avg_total_delay] >= delay_length)]
        grouped_df = grouped_df.dropna()
        return grouped_df

    def get_boxplot_data(self, selected_states, selected_months, selected_airlines, selected_days, show_outliers):
        """ function to produce boxplot carrier and delay tab """
        df = self.flight.copy().reset_index()

        # filter for selected states, months, carriers, days of week
        filtered_df = df[
            (df["ORIGIN_STATE"].isin(selected_states)) &
            (df["MONTH"].isin(selected_months)) &
            (df["OP_UNIQUE_CARRIER"].isin(selected_airlines)) &
            (df["DAY_OF_WEEK"].map(DAY_OF_WEEK_MAPPING).isin(selected_days))
            ]

        filtered_df = filtered_df[filtered_df["CAUSE_OF_DELAY"] != "None"].copy()

        return filtered_df

    def get_delay_types(self):
        """ function to get delay types from flight_df """
        delays = self.flight['CAUSE_OF_DELAY'].unique()
        return [delay for delay in delays if delay != 'None']
