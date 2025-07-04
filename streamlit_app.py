import os
import time
import requests
import pandas as pd
import googlemaps
from haversine import haversine, Unit
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import quote_plus

# --- Page and API Key Setup ---
st.set_page_config(layout="wide")
st.title("📍 Business Sequence Finder")

load_dotenv()
API_KEY = os.getenv("GMAPS_API_KEY")

if not API_KEY:
    st.error("GMAPS_API_KEY not found in .env file. Please add it and restart.")
    st.stop()

try:
    gmaps = googlemaps.Client(key=API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Google Maps client. Check your API key. Error: {e}")
    st.stop()


# --- Data and Functions ---
PLACE_TYPES = [
  "car_dealer","car_rental","car_repair","car_wash",
  "electric_vehicle_charging_station","gas_station","parking","rest_stop",
  "art_gallery","museum","performing_arts_theater",
  "library","school","university",
  "amusement_park","aquarium","bowling_alley",
  "casino","community_center","movie_theater","night_club",
  "park","zoo",
  "atm","bank",
  "bakery","bar","cafe","restaurant",
  "hospital","drugstore","pharmacy",
  "hotel","motel",
  "church","hindu_temple","mosque","synagogue",
  "barber_shop","beauty_salon","hair_salon",
  "laundry","lawyer",
  "book_store","clothing_store",
  "convenience_store","grocery_store","supermarket",
  "fitness_center","gym","stadium",
  "bus_station","train_station","taxi_stand"
]

@st.cache_data
def get_autocomplete_suggestions(input_text):
    if not input_text:
        return []
    try:
        response = gmaps.places_autocomplete(
            input_text,
            types='address',
            components={"country": "us"}
        )
        return response
    except Exception as e:
        st.error(f"Autocomplete API error: {e}")
        return []

@st.cache_data
def get_details_from_place_id(place_id):
    if not place_id:
        return None, None
    try:
        detail = gmaps.place(
            place_id=place_id,
            fields=["address_component", "formatted_address"]
        ).get("result", {})

        if not detail:
            return None, None

        comps = detail.get("address_components", [])
        zip_code = next((c["short_name"] for c in comps if "postal_code" in c["types"]), "")
        full_address = detail.get("formatted_address", "")

        return full_address, zip_code
    except Exception as e:
        st.error(f"Place Details API error: {e}")
        return None, None

@st.cache_data(ttl=600)
def fetch_places(_gmaps_client, center_coords, radius_miles, place_types, _progress_bar, target_zip=None, per_type=60):
    found = {}
    total_types = len(place_types)

    for i, ptype in enumerate(place_types):
        progress_percentage = (i + 1) / total_types
        progress_text = f"Searching for: {ptype.replace('_', ' ').title()}..."
        _progress_bar.progress(progress_percentage, text=progress_text)

        page_token = None
        count = 0
        while True:
            try:
                radius_meters = int(radius_miles * 1609.34)
                resp = gmaps.places_nearby(
                    location=center_coords,
                    radius=radius_meters,
                    type=ptype,
                    page_token=page_token
                )
            except Exception as e:
                st.error(f"Places Nearby API error for type '{ptype}': {e}")
                break

            for p in resp.get("results", []):
                pid = p.get("place_id")
                if pid and pid not in found:
                    loc = p["geometry"]["location"]
                    found[pid] = {
                        "place_id": pid,
                        "name": p.get("name"),
                        "address": p.get("vicinity"),
                        "latlng": (loc["lat"], loc["lng"])
                    }
                count += 1
                if count >= per_type:
                    break

            page_token = resp.get("next_page_token")
            if not page_token or count >= per_type:
                break
            time.sleep(2)

    _progress_bar.progress(1.0, text="Fetching business details...")
    rows = []
    for biz in found.values():
        detail = gmaps.place(
            place_id=biz["place_id"],
            fields=["formatted_phone_number", "address_component"]
        ).get("result", {})

        zipc = ""
        if detail:
            comps = detail.get("address_components", [])
            zipc = next((c["short_name"] for c in comps if "postal_code" in c["types"]), "")

        if not target_zip or (target_zip and zipc == target_zip):
            rows.append({
                "name": biz["name"],
                "address": biz["address"],
                "phone": detail.get("formatted_phone_number", ""),
                "zip_code": zipc,
                "lat": biz["latlng"][0],
                "lon": biz["latlng"][1]
            })

    df = pd.DataFrame(rows)
    _progress_bar.empty()
    return df

def order_by_nearest(df, start_coords):
    if len(df) < 2:
        return df

    df2 = df.copy()
    df2["dist_from_start"] = df2.apply(
        lambda r: haversine(start_coords, (r.lat, r.lon), unit=Unit.MILES),
        axis=1
    )
    far_idx = df2["dist_from_start"].idxmax()
    farthest = df2.loc[far_idx]
    remaining = df2.drop(index=far_idx).copy()

    sequence = []
    current = start_coords
    while not remaining.empty:
        remaining["temp_dist"] = remaining.apply(
            lambda r: haversine(current, (r.lat, r.lon), unit=Unit.MILES),
            axis=1
        )
        next_idx = remaining["temp_dist"].idxmin()
        next_row = remaining.loc[next_idx]
        sequence.append(next_row)
        current = (next_row.lat, next_row.lon)
        remaining = remaining.drop(index=next_idx)

    sequence.append(farthest)
    ordered = pd.DataFrame(sequence).reset_index(drop=True)
    return ordered.drop(columns=["dist_from_start","temp_dist"], errors="ignore")


# --- Streamlit UI and Logic ---

if "address" not in st.session_state:
    st.session_state.address = "The White House, 1600 Pennsylvania Ave NW, Washington, DC 20500"
    st.session_state.zip_code = "20500"
    st.session_state.search_input = ""
    st.session_state.final_df = None

# Step 1: Set a Starting Point
st.subheader("1. Set a Starting Point")
st.info(f"**Current starting address:** {st.session_state.address}")
st.session_state.search_input = st.text_input(
    "Search for a new address",
    value=st.session_state.search_input,
    placeholder="Type an address and press Enter..."
)

if st.session_state.search_input:
    suggestions = get_autocomplete_suggestions(st.session_state.search_input)
    if suggestions:
        suggestion_map = {pred["description"]: pred["place_id"] for pred in suggestions}

        selected_suggestion = st.selectbox(
            "Select the correct address:",
            options=[""] + list(suggestion_map.keys()),
            format_func=lambda x: "Please choose an option..." if x == "" else x,
            key="suggestion_box"
        )

        if selected_suggestion:
            place_id = suggestion_map[selected_suggestion]
            full_address, zip_code = get_details_from_place_id(place_id)

            st.session_state.address = full_address
            st.session_state.zip_code = zip_code
            st.session_state.search_input = ""
            st.session_state.final_df = None
            st.rerun()

st.markdown("---")

# Step 2: Define Search Area and Business Types
st.subheader("2. Define Search Area and Business Types")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Choose search method:**")
    search_mode = st.radio(
        "Search Mode",
        ("Radius from starting point", "Entire ZIP code territory"),
        label_visibility="collapsed"
    )

    if search_mode == "Radius from starting point":
        radius = st.slider("Search Radius (miles)", 1, 15, 3)
    else:
        st.info(f"Will search the entire territory of ZIP Code: **{st.session_state.get('zip_code', 'N/A')}**")

with col2:
    st.write("**Select business types to include:**")
    select_all = st.checkbox("Select all business types")

    default_selection = []
    if select_all:
        default_selection = PLACE_TYPES
    else:
        default_selection = ["restaurant", "cafe", "bar", "grocery_store"]

    place_types = st.multiselect(
        "Place Types",
        options=PLACE_TYPES,
        default=default_selection,
        label_visibility="collapsed"
    )

st.markdown("---")

# Step 3: Generate the Sequence
if st.button("🚀 Fetch and Sequence Places", use_container_width=True):
    start_coords = None
    target_zip_for_search = None
    radius_for_search = 6

    if search_mode == "Entire ZIP code territory":
        current_zip = st.session_state.get('zip_code')
        if not current_zip:
            st.error("No ZIP code available. Please select a starting address with a valid ZIP code.")
            st.stop()
        with st.spinner(f"Geocoding ZIP code {current_zip}..."):
            geo = gmaps.geocode(current_zip)
        if not geo:
            st.error(f"Could not find a location for ZIP code {current_zip}.")
            st.stop()

        loc = geo[0]["geometry"]["location"]
        start_coords = (loc["lat"], loc["lng"])
        target_zip_for_search = current_zip
        st.success(f"Set search center to the middle of ZIP Code {target_zip_for_search}")

    else:
        if not st.session_state.address:
            st.error("Please enter a starting address.")
            st.stop()
        with st.spinner(f"Geocoding address: {st.session_state.address}..."):
            geo = gmaps.geocode(st.session_state.address)
        if not geo:
            st.error(f"Could not geocode the address: {st.session_state.address}")
            st.stop()

        loc = geo[0]["geometry"]["location"]
        start_coords = (loc["lat"], loc["lng"])
        radius_for_search = radius
        target_zip_for_search = None
        st.success(f"Set search center to the selected address.")

    progress_bar = st.progress(0, text="Initializing search...")
    df = fetch_places(
        gmaps,
        center_coords=start_coords,
        radius_miles=radius_for_search,
        place_types=place_types,
        _progress_bar=progress_bar,
        target_zip=target_zip_for_search
    )

    if df.empty:
        st.warning("No places found. Try expanding the radius/ZIP or changing business types.")
        st.session_state.final_df = None
    else:
        with st.spinner("Geocoding final start point for sequencing..."):
            geo = gmaps.geocode(st.session_state.address)

        if not geo:
            st.error("Could not geocode the starting address for final sequencing.")
            st.stop()

        actual_start_coords = (geo[0]["geometry"]["location"]["lat"], geo[0]["geometry"]["location"]["lng"])
        ordered_df = order_by_nearest(df, actual_start_coords)

        ordered_df['next_lat'] = ordered_df['lat'].shift(-1)
        ordered_df['next_lon'] = ordered_df['lon'].shift(-1)

        def calc_dist_to_next(row):
            if pd.isna(row['next_lat']):
                return None
            return round(haversine((row['lat'], row['lon']), (row['next_lat'], row['next_lon']), unit=Unit.MILES), 2)

        ordered_df['Distance to Next (miles)'] = ordered_df.apply(calc_dist_to_next, axis=1)

        final_df = ordered_df[['name', 'address', 'phone', 'zip_code', 'Distance to Next (miles)', 'lat', 'lon']].copy()
        st.session_state.final_df = final_df
        # Rerun to clear the button press and display the table
        st.rerun()

if st.session_state.get('final_df') is not None:
    final_df = st.session_state.final_df
    st.success(f"Found {len(final_df)} places and created a route!")
    st.map(final_df[["lat", "lon"]])

    st.markdown("---")
    st.subheader("4. Create Route on Google Maps")
    st.info("Select 2 to 10 stops from the table below to generate a route. The first selected stop will be the origin.")

    # Display the interactive dataframe
    edited_df = st.dataframe(
        final_df[['name', 'address', 'phone', 'zip_code', 'Distance to Next (miles)']],
        on_select="rerun",
        selection_mode="multi-row",
        key='df_select' # This key is managed by Streamlit
    )

    # Access the selection from the widget's state
    selected_indices = edited_df.selection.rows
    num_selected = len(selected_indices)

    if num_selected > 10:
        st.warning("You can select a maximum of 10 stops.")
    elif num_selected >= 2:
        selected_rows = final_df.iloc[selected_indices]

        # Use the full, detailed address from the original dataframe for accuracy
        # The 'address' column from Google is usually more robust than 'vicinity'
        full_addresses = []
        with st.spinner("Verifying addresses for route..."):
            for index, row in selected_rows.iterrows():
                # Re-geocoding the address ensures the most accurate, routable format
                detail = gmaps.place(place_id=gmaps.find_place(row['name'] + ' ' + row['address'], 'textquery')['candidates'][0]['place_id'], fields=['formatted_address']).get('result', {})
                if detail and 'formatted_address' in detail:
                    full_addresses.append(detail['formatted_address'])
                else:
                    # Fallback to the known address if detail fetch fails
                    full_addresses.append(row['address'])
        
        encoded_addresses = [quote_plus(addr) for addr in full_addresses]

        # A more robust URL format for directions
        origin = encoded_addresses[0]
        destination = encoded_addresses[-1]
        waypoints = '|'.join(encoded_addresses[1:-1])

        if waypoints:
            maps_url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&waypoints={waypoints}"
        else:
            maps_url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"


        st.link_button("Click here to Open Route in Google Maps", url=maps_url, use_container_width=True)

    csv_bytes = final_df[['name', 'address', 'phone', 'zip_code', 'Distance to Next (miles)']].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full List as CSV",
        data=csv_bytes,
        file_name="business_sequence.csv",
        mime="text/csv",
        use_container_width=True
    )