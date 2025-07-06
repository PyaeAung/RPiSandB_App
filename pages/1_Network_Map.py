import streamlit as st
from obspy.clients.fdsn import Client
import folium
from streamlit_folium import st_folium
import pandas as pd

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="R-Shake Network Map")
st.title("Raspberry Shake Station Network Map")
st.markdown("A map showing the locations of specified Raspberry Shake stations.")

# --- Station List ---
# List of station codes you provided
STATION_CODES = [
    "S9164", "R29DA", "R7183", "R237D", "RCEEF",
    "RDBC5", "RB2DA", "R7107", "R262E", "RFB8B"
]
NETWORK_CODE = "AM" # Assuming all are from the AM network

# --- Data Fetching with Caching ---
# @st.cache_data decorator will cache the output of this function.
# When the app is re-run, if the inputs haven't changed, Streamlit
# will use the cached data instead of fetching it again.
@st.cache_data
def get_station_metadata(network, stations):
    """
    Fetches metadata for a list of stations and returns it as a pandas DataFrame.
    """
    client = Client("https://data.raspberryshake.org/")
    station_data = []
    
    # We will fetch metadata for all stations at once for efficiency
    try:
        inventory = client.get_stations(network=network, station=",".join(stations), level="channel")
        
        for net in inventory:
            for sta in net:
                # Identify station type (1D, 3D, 4D)
                channels = {ch.code for ch in sta.channels}
                station_type = "Unknown"
                if "HDF" in channels:
                    station_type = "4D (Shake & Boom)"
                elif "ENE" in channels or "ENN" in channels:
                    station_type = "3D Shake"
                elif "EHZ" in channels or "SHZ" in channels:
                    station_type = "1D Shake"

                station_data.append({
                    "code": sta.code,
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "elevation_m": sta.elevation,
                    "type": station_type,
                    "description": sta.site.name if sta.site.name else "N/A"
                })
        return pd.DataFrame(station_data)
        
    except Exception as e:
        # Return an empty DataFrame if there's an error
        st.error(f"Could not fetch station data: {e}")
        return pd.DataFrame()

# --- Main App Logic ---
st.write("Fetching metadata for the following stations:")
st.write(f"`{', '.join(STATION_CODES)}`")

# Fetch the data
df_stations = get_station_metadata(NETWORK_CODE, STATION_CODES)

if not df_stations.empty:
    st.success(f"Successfully fetched metadata for {len(df_stations)} stations.")
    
    # --- Create the Map ---
    # Calculate the center of the map
    map_center = [df_stations['latitude'].mean(), df_stations['longitude'].mean()]
    
    # Initialize the map with Folium
    m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")

    # Add markers for each station
    for idx, row in df_stations.iterrows():
        # Define marker color based on station type
        color = "blue" # Default
        if "4D" in row['type']:
            color = "red"
        elif "3D" in row['type']:
            color = "orange"
        elif "1D" in row['type']:
            color = "green"

        # Create the HTML for the popup
        popup_html = f"""
        <b>Station:</b> {row['code']}<br>
        <b>Type:</b> {row['type']}<br>
        <b>Description:</b> {row['description']}<br>
        <b>Coordinates:</b> ({row['latitude']:.4f}, {row['longitude']:.4f})
        """
        
        # Create a Folium popup
        popup = folium.Popup(popup_html, max_width=300)
        
        # Add the marker to the map
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            tooltip=f"{row['code']} ({row['type']})", # Shows on hover
            icon=folium.Icon(color=color, icon="server", prefix='fa')
        ).add_to(m)
        
    # --- Display the Map in Streamlit ---
    st_folium(m, width=1200, height=700, returned_objects=[])

    # --- Display Station Data as a Table ---
    st.subheader("Station Details")
    st.dataframe(df_stations)

else:
    st.warning("No station data could be displayed. Please check the station codes or network connectivity.")