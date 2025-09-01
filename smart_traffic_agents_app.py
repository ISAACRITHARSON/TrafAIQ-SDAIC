import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import osv
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Smart Traffic Management System",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
        box-shadow: 0 0.25rem 0.5rem rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #e3f2fd;
        border-radius: 0.3rem;
        padding: 0.5rem;
        text-align: center;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================= LLM INTEGRATION =========================

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    else:
        return st.session_state.get("api_key", "")

def set_openai_client():
    api_key = get_api_key()
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        return client
    return None

def call_llm(prompt, system_message="You are an AI traffic management expert. Provide concise, accurate responses."):
    """Call OpenAI API with the given prompt"""
    client = set_openai_client()
    
    if not client:
        return "API key not set. Please enter your OpenAI API key in the sidebar."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

# ========================= AI AGENTS =========================

def analyze_traffic_congestion(df, street_name=None, use_llm=True):
    """AI Agent to analyze traffic congestion levels"""
    # Filter by street name if provided
    if street_name and street_name != "All Streets":
        df = df[df['Street_Name'] == street_name]
    
    if not use_llm:
        # Generate mock response if LLM is not used
        return {
            "congested_locations": [
                {"location": df['Street_Name'].iloc[0], "severity": 8},
                {"location": df['Street_Name'].iloc[1], "severity": 7},
                {"location": df['Street_Name'].iloc[2], "severity": 6}
            ]
        }
    
    # Get congestion data by location
    location_stats = df.groupby(['Street_Name', 'Latitude', 'Longitude']).agg({
        'Count': 'sum', 
        'Accident': 'sum', 
        'Road_Block': 'sum',
        'Traffic_Level': lambda x: list(x)
    }).reset_index()
    
    # Adjust the prompt based on whether we're analyzing a specific street or all streets
    if street_name and street_name != "All Streets":
        prompt = f"""
        You are a traffic AI agent analyzing congestion levels for {street_name}. Based on the following data:
        
        {location_stats[['Street_Name', 'Count', 'Accident', 'Road_Block']].to_string(index=False)}
        
        Return a detailed analysis of the congestion at this location with severity score (1-10) in JSON format:
        {{
            "congested_locations": [
                {{"location": "{street_name}", "severity": score}}
            ]
        }}
        ONLY return the JSON, nothing else.
        """
    else:
        prompt = f"""
        You are a traffic AI agent analyzing congestion levels. Based on the following data:
        
        {location_stats[['Street_Name', 'Count', 'Accident', 'Road_Block']].head(10).to_string(index=False)}
        
        Return the top 3 most congested locations with severity scores (1-10) in JSON format:
        {{
            "congested_locations": [
                {{"location": "Street Name", "severity": score}},
                ...
            ]
        }}
        ONLY return the JSON, nothing else.
        """
    
    try:
        result = call_llm(prompt)
        # Clean and parse the result
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing congestion analysis: {str(e)}")
        return {
            "congested_locations": [
                {"location": "Error in analysis", "severity": 5}
            ]
        }

def suggest_traffic_signals(df, street_name=None, use_llm=True):
    """AI Agent to suggest traffic signal status based on congestion"""
    # Filter by street name if provided
    if street_name and street_name != "All Streets":
        df = df[df['Street_Name'] == street_name]
    
    if not use_llm:
        # Generate mock response if LLM is not used
        signals = []
        for _, row in df.head(5).iterrows():
            if row['Count'] > 30:
                signal = "red"
            elif row['Count'] > 15:
                signal = "yellow"
            else:
                signal = "green"
            signals.append({
                "location": row['Street_Name'],
                "lat": row['Latitude'],
                "lon": row['Longitude'],
                "signal": signal,
                "vehicles": int(row['Count'])
            })
        return {"signals": signals}
    
    # Get traffic data by unique locations
    signal_input = df.groupby(['Street_Name', 'Latitude', 'Longitude']).agg({
        'Count': 'sum', 
        'Accident': 'sum', 
        'Road_Block': 'sum'
    }).reset_index()
    
    # Adjust prompt based on whether we're analyzing a specific street
    if street_name and street_name != "All Streets":
        prompt = f"""
        You are a smart traffic light controller for {street_name}. Based on the following rules:
        - Vehicle count > 30 = Red signal
        - Vehicle count between 15-30 = Yellow signal
        - Vehicle count < 15 = Green signal
        - Any location with an accident or road block should be Red
        
        Data:
        {signal_input[['Street_Name', 'Count', 'Accident', 'Road_Block']].to_string(index=False)}
        
        Return a JSON with the suggested signal:
        {{
            "signals": [
                {{
                    "location": "{street_name}",
                    "lat": {signal_input['Latitude'].iloc[0] if len(signal_input) > 0 else 0},
                    "lon": {signal_input['Longitude'].iloc[0] if len(signal_input) > 0 else 0},
                    "signal": "red/yellow/green",
                    "vehicles": count
                }}
            ]
        }}
        ONLY return the JSON, nothing else.
        """
    else:
        prompt = f"""
        You are a smart traffic light controller. Based on the following rules:
        - Vehicle count > 30 = Red signal
        - Vehicle count between 15-30 = Yellow signal
        - Vehicle count < 15 = Green signal
        - Any location with an accident or road block should be Red
        
        Data:
        {signal_input[['Street_Name', 'Count', 'Accident', 'Road_Block']].head(10).to_string(index=False)}
        
        Return a JSON list with locations and their suggested signals:
        {{
            "signals": [
                {{
                    "location": "Street Name",
                    "lat": latitude,
                    "lon": longitude,
                    "signal": "red/yellow/green",
                    "vehicles": count
                }},
                ...
            ]
        }}
        ONLY return the JSON, nothing else.
        """
    
    try:
        result = call_llm(prompt)
        # Clean and parse the result
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing traffic signal suggestions: {str(e)}")
        return {"signals": []}

def classify_rush_hours(df, street_name=None, use_llm=True):
    """AI Agent to classify hours as rush hour, moderate, or light traffic"""
    # Filter by street name if provided
    if street_name and street_name != "All Streets":
        df = df[df['Street_Name'] == street_name]
    
    if not use_llm:
        # Generate mock response if LLM is not used
        return {
            "hour_classification": {
                "07": "Rush Hour",
                "08": "Rush Hour",
                "09": "Rush Hour",
                "16": "Rush Hour",
                "17": "Rush Hour",
                "18": "Rush Hour",
                "12": "Moderate",
                "13": "Moderate",
                "10": "Light",
                "14": "Light"
            }
        }
    
    # Extract hour from time strings and group
    df['Hour'] = df['Time_of_Day'].str.split(':').str[0]
    hourly_stats = df.groupby('Hour').agg({
        'Count': ['mean', 'sum', 'count'],
        'Traffic_Level': lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
    }).reset_index()
    
    # Adjust prompt based on whether we're analyzing a specific street
    street_context = f" for {street_name}" if street_name and street_name != "All Streets" else ""
    prompt = f"""
    You are a traffic analyst AI. Classify hours as 'Rush Hour', 'Moderate', or 'Light' based on vehicle count{street_context}.
    
    Data:
    {hourly_stats.to_string(index=False)}
    
    Return a JSON mapping like:
    {{
        "hour_classification": {{
            "07": "Rush Hour",
            "08": "Rush Hour",
            "12": "Moderate",
            "22": "Light",
            ...
        }}
    }}
    Include all 24 hours (00-23). ONLY return the JSON, nothing else.
    """
    
    try:
        result = call_llm(prompt)
        # Clean and parse the result
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing rush hour classification: {str(e)}")
        return {"hour_classification": {}}

def suggest_alternative_routes(df, street_name=None, use_llm=True):
    """AI Agent to suggest alternative routes for congested streets"""
    # Filter by street name if provided
    if street_name and street_name != "All Streets":
        df = df[df['Street_Name'] == street_name]
    
    if not use_llm:
        # Generate mock response if LLM is not used
        return {
            "route_suggestions": [
                {
                    "congested_street": df['Street_Name'].iloc[0],
                    "alternative_route": df['Alternative_Route'].iloc[0],
                    "current_traffic": "Heavy",
                    "estimated_time_saved": "12 minutes"
                },
                {
                    "congested_street": df['Street_Name'].iloc[1],
                    "alternative_route": df['Alternative_Route'].iloc[1],
                    "current_traffic": "Moderate",
                    "estimated_time_saved": "8 minutes"
                }
            ]
        }
    
    # Get congested streets
    congested = df[df['Traffic_Level'].isin(['Rush Hour', 'Abnormal'])].groupby(['Street_Name', 'Alternative_Route']).agg({
        'Count': 'sum',
        'Traffic_Level': lambda x: list(x),
        'Accident': 'sum',
        'Road_Block': 'sum'
    }).reset_index()
    
    if len(congested) == 0:
        return {"route_suggestions": []}
    
    # Adjust prompt based on whether we're analyzing a specific street
    if street_name and street_name != "All Streets":
        prompt = f"""
        You are a routing agent suggesting alternative routes for {street_name}.
        
        Data on congestion:
        {congested.to_string(index=False)}
        
        Return suggestion in JSON format:
        {{
            "route_suggestions": [
                {{
                    "congested_street": "{street_name}",
                    "alternative_route": "Alternative Route",
                    "current_traffic": "Heavy/Moderate/Light",
                    "estimated_time_saved": "X minutes"
                }}
            ]
        }}
        Estimate time saved based on congestion levels, accidents, and road blocks.
        ONLY return the JSON, nothing else.
        """
    else:
        prompt = f"""
        You are a routing agent suggesting alternative routes for congested streets.
        
        Data on congested streets:
        {congested.to_string(index=False)}
        
        Return suggestions in JSON format:
        {{
            "route_suggestions": [
                {{
                    "congested_street": "Street Name",
                    "alternative_route": "Alternative Route",
                    "current_traffic": "Heavy/Moderate/Light",
                    "estimated_time_saved": "X minutes"
                }},
                ...
            ]
        }}
        Estimate time saved based on congestion levels, accidents, and road blocks.
        ONLY return the JSON, nothing else.
        """
    
    try:
        result = call_llm(prompt)
        # Clean and parse the result
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing route suggestions: {str(e)}")
        return {"route_suggestions": []}

def adjust_dynamic_tolls(df, street_name=None, use_llm=True):
    """AI Agent to adjust toll prices based on congestion levels"""
    # Filter by street name if provided
    if street_name and street_name != "All Streets":
        df = df[df['Street_Name'] == street_name]
    
    if not use_llm:
        # Generate mock response if LLM is not used
        return {
            "toll_adjustments": [
                {
                    "route": df['Suggested_Route'].iloc[0],
                    "base_toll": float(df['Toll_Price'].iloc[0]),
                    "adjusted_toll": float(df['Toll_Price'].iloc[0]) * 1.5,
                    "reason": "Rush hour congestion"
                },
                {
                    "route": df['Suggested_Route'].iloc[1],
                    "base_toll": float(df['Toll_Price'].iloc[1]),
                    "adjusted_toll": float(df['Toll_Price'].iloc[1]),
                    "reason": "Normal traffic conditions"
                }
            ]
        }
    
    # Get toll data by route
    toll_input = df.groupby('Suggested_Route').agg({
        'Toll_Price': 'mean',
        'Traffic_Level': lambda x: list(x),
        'Count': 'sum',
        'Accident': 'sum',
        'Road_Block': 'sum'
    }).reset_index()
    
    # Adjust prompt based on whether we're analyzing a specific street
    street_context = f" for routes near {street_name}" if street_name and street_name != "All Streets" else ""
    prompt = f"""
    You are an AI adjusting dynamic tolls based on traffic conditions{street_context}. 
    General rules:
    - Increase by 50% during Rush Hour
    - Increase by 100% for Abnormal traffic
    - No change for Light or Moderate traffic
    - Additional 25% increase if accidents are present
    
    Data:
    {toll_input.to_string(index=False)}
    
    Return JSON with toll adjustments:
    {{
        "toll_adjustments": [
            {{
                "route": "Route Name",
                "base_toll": original_toll,
                "adjusted_toll": new_toll,
                "reason": "Explanation for adjustment"
            }},
            ...
        ]
    }}
    ONLY return the JSON, nothing else.
    """
    
    try:
        result = call_llm(prompt)
        # Clean and parse the result
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing toll adjustments: {str(e)}")
        return {"toll_adjustments": []}

# ========================= UI COMPONENTS =========================

def show_map(df, signals=None):
    """Display interactive map with traffic data"""
    # Color mapping based on traffic level
    color_mapping = {
        'Light': [0, 255, 0, 160],      # Green
        'Moderate': [255, 255, 0, 160], # Yellow
        'Rush Hour': [255, 165, 0, 160],# Orange
        'Abnormal': [255, 0, 0, 160]    # Red
    }
    
    # Prepare map data
    map_data = df.copy()
    map_data['color'] = map_data['Traffic_Level'].map(lambda x: color_mapping.get(x, [100, 100, 100, 160]))
    
    # Configure the map view
    view_state = pdk.ViewState(
        latitude=map_data['Latitude'].mean(),
        longitude=map_data['Longitude'].mean(),
        zoom=2,
        pitch=0
    )
    
    # Create the scatter plot layer
    traffic_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=['Longitude', 'Latitude'],
        get_color='color',
        get_radius='Count / 2',  # Size based on vehicle count
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_min_pixels=5,
        radius_max_pixels=30,
    )
    
    # Add traffic signal layer if available
    layers = [traffic_layer]
    
    if signals and 'signals' in signals:
        signal_data = []
        for signal in signals['signals']:
            try:
                # Color based on signal status
                if signal['signal'] == 'red':
                    color = [255, 0, 0, 200]
                elif signal['signal'] == 'yellow':
                    color = [255, 255, 0, 200]
                else:  # green
                    color = [0, 255, 0, 200]
                
                signal_data.append({
                    'lat': signal['lat'],
                    'lon': signal['lon'],
                    'color': color,
                    'location': signal['location'],
                    'signal': signal['signal'],
                    'vehicles': signal['vehicles']
                })
            except KeyError:
                continue
        
        if signal_data:
            signal_df = pd.DataFrame(signal_data)
            
            signal_layer = pdk.Layer(
                "ScatterplotLayer",
                data=signal_df,
                get_position=['lon', 'lat'],
                get_color='color',
                get_radius=25,
                pickable=True,
                opacity=1.0,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
            )
            
            layers.append(signal_layer)
    
    # Create the tooltip
    tooltip = {
        "html": "<b>{Street_Name}</b><br/>"
                "Vehicle Type: {Vehicle_Type}<br/>"
                "Count: {Count}<br/>"
                "Traffic: {Traffic_Level}<br/>"
                "Time: {Time_of_Day}",
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"}
    }
    
    # Create the deck
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip
    )
    
    return r

def show_vehicle_distribution(filtered_df, location=None):
    """Show vehicle type distribution for selected location"""
    if location and location != "All Streets":
        location_data = filtered_df[filtered_df['Street_Name'] == location]
    else:
        location_data = filtered_df
    
    # Aggregate vehicle counts by type
    vehicle_counts = location_data.groupby('Vehicle_Type')['Count'].sum().reset_index()
    
    # Create bar chart
    fig = px.bar(
        vehicle_counts,
        x='Vehicle_Type',
        y='Count',
        color='Vehicle_Type',
        title=f"Vehicle Distribution {f'at {location}' if location and location != 'All Streets' else 'Overall'}"
    )
    
    return fig

def show_hourly_distribution(filtered_df, location=None):
    """Show traffic distribution by hour of day"""
    # Filter by location if specified
    if location and location != "All Streets":
        filtered_df = filtered_df[filtered_df['Street_Name'] == location]
    
    # Extract hour from time
    filtered_df['Hour'] = filtered_df['Time_of_Day'].str.split(':').str[0]
    hourly_counts = filtered_df.groupby('Hour')['Count'].sum().reset_index()
    
    # Sort by hour
    hourly_counts['Hour'] = hourly_counts['Hour'].astype(int)
    hourly_counts = hourly_counts.sort_values('Hour')
    hourly_counts['Hour'] = hourly_counts['Hour'].astype(str).str.zfill(2)
    
    # Create bar chart
    fig = px.bar(
        hourly_counts,
        x='Hour',
        y='Count',
        title=f"Traffic Volume by Hour of Day {f'at {location}' if location and location != 'All Streets' else 'Overall'}",
        labels={'Hour': 'Hour of Day', 'Count': 'Total Vehicle Count'}
    )
    
    return fig

# ========================= MAIN APPLICATION =========================

def main():
    st.markdown("<h1 class='main-header'>Multi-Agent System for Smart Traffic Management</h1>", unsafe_allow_html=True)
    
    # Sidebar for API key
    st.sidebar.title("OpenAI API Configuration")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key
    
    # Option to use LLM or not
    use_llm = st.sidebar.checkbox("Use OpenAI for AI Agents", value=True)
    if use_llm and not get_api_key():
        st.sidebar.warning("Please enter an OpenAI API key to use AI agents")
        use_llm = False
    
    # Upload or generate data
    st.markdown("<h2 class='sub-header'>Upload or Generate Traffic Data</h2>", unsafe_allow_html=True)
    
    data_option = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with traffic data", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
        else:
            st.info("Please upload a CSV file to continue.")
            return
    else:
        try:
            df = pd.read_csv('traffic_data.csv')
            st.success("Sample data loaded successfully!")
        except FileNotFoundError:
            st.error("Sample data file not found. Please run the data generation script first.")
            if st.button("Generate Sample Data"):
                # This is a placeholder - in a real app, you'd run the generation script
                st.info("This would generate sample data in a real app.")
                st.stop()
            return
    
    # Display basic data statistics
    st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Locations", df['Street_Name'].nunique())
    with col3:
        st.metric("Vehicle Types", df['Vehicle_Type'].nunique())
    with col4:
        st.metric("Average Vehicles", int(df['Count'].mean()))
    
    # Filters
    st.markdown("<h2 class='sub-header'>Filter Data</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Changed from vehicle types to street names
        selected_street = st.selectbox(
            "Street Name",
            ["All Streets"] + sorted(df['Street_Name'].unique().tolist())
        )
    
    with col2:
        selected_traffic_levels = st.multiselect(
            "Traffic Levels",
            df['Traffic_Level'].unique(),
            default=df['Traffic_Level'].unique()
        )
    
    with col3:
        # Extract hours for filtering
        df['Hour'] = df['Time_of_Day'].str.split(':').str[0].astype(int)
        hour_range = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=(0, 23)
        )
    
    # Apply filters
    if selected_street != "All Streets":
        filtered_df = df[
            (df['Street_Name'] == selected_street) &
            (df['Traffic_Level'].isin(selected_traffic_levels)) &
            (df['Hour'] >= hour_range[0]) &
            (df['Hour'] <= hour_range[1])
        ]
    else:
        filtered_df = df[
            (df['Traffic_Level'].isin(selected_traffic_levels)) &
            (df['Hour'] >= hour_range[0]) &
            (df['Hour'] <= hour_range[1])
        ]
    
    # Show AI Agent results in tabs
    st.markdown("<h2 class='sub-header'>AI Traffic Management Agents</h2>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Traffic Map", 
        "🚦 Traffic Signals", 
        "🕒 Rush Hour Analysis", 
        "🔀 Alternative Routes", 
        "💰 Dynamic Tolls",
        "📊 Traffic Statistics"
    ])
    
    # Tab 1: Traffic Map with vehicle distribution
    with tab1:
        st.markdown("<h3>Interactive Traffic Map</h3>", unsafe_allow_html=True)
        st.write(f"Explore the traffic patterns {f'at {selected_street}' if selected_street != 'All Streets' else 'across different locations'}. Click on points to view details.")
        
        # Run traffic signal agent to get signals for the map
        with st.spinner("Analyzing traffic signals..."):
            traffic_signals = suggest_traffic_signals(filtered_df, selected_street, use_llm)
        
        # Display map
        map_chart = show_map(filtered_df, traffic_signals)
        st.pydeck_chart(map_chart)
        
        # Vehicle distribution for the selected location
        st.markdown("<h3>Vehicle Distribution</h3>", unsafe_allow_html=True)
        vehicle_chart = show_vehicle_distribution(filtered_df, selected_street)
        st.plotly_chart(vehicle_chart, use_container_width=True)
        # Tab 2: Smart Traffic Signal Agent
    with tab2:
        st.markdown("<h3>Smart Traffic Signal Agent</h3>", unsafe_allow_html=True)
        st.write(f"This agent analyzes traffic patterns {f'at {selected_street}' if selected_street != 'All Streets' else 'across all locations'} and suggests optimal traffic signal settings.")
    
        # Display signal recommendations
        if 'signals' in traffic_signals:
            st.write("Recommended Traffic Signal Status:")
            cols = st.columns(3)
        for i, signal in enumerate(traffic_signals['signals']):
            col_idx = i % 3
            with cols[col_idx]:
                signal_color = signal['signal']
                bg_color = "#f44336" if signal_color == "red" else "#ffeb3b" if signal_color == "yellow" else "#4caf50"
                text_color = "white" if signal_color == "red" else "black"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4 style="margin: 0;">{signal['location']}</h4>
                    <p style="margin: 5px 0;">Signal: {signal['signal'].upper()}</p>
                    <p style="margin: 5px 0;">Vehicles: {signal['vehicles']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No traffic signal data available for the selected street.")

# Tab 3: Rush Hour Analysis
    with tab3:
        st.markdown("<h3>Rush Hour Analysis</h3>", unsafe_allow_html=True)
        st.write(f"This agent analyzes traffic patterns to identify rush hours {f'at {selected_street}' if selected_street != 'All Streets' else 'across all locations'}.")

        with st.spinner("Classifying rush hours..."):
            rush_hour_info = classify_rush_hours(filtered_df, selected_street, use_llm)

        if 'hour_classification' in rush_hour_info:
            rush_data = rush_hour_info['hour_classification']
            rush_df = pd.DataFrame(list(rush_data.items()), columns=['Hour', 'Traffic Category'])
            st.dataframe(rush_df)
            fig_rush = px.bar(rush_df, x='Hour', y='Traffic Category', color='Traffic Category',
                            category_orders={'Hour': [str(i).zfill(2) for i in range(24)]},
                            title="Hourly Traffic Classification")
            st.plotly_chart(fig_rush, use_container_width=True)
        else:
            st.warning("No rush hour classification data available.")

    # Tab 4: Alternative Routes
    with tab4:
        st.markdown("<h3>Alternative Route Suggestions</h3>", unsafe_allow_html=True)

        with st.spinner("Analyzing alternative routes..."):
            alt_routes = suggest_alternative_routes(filtered_df, selected_street, use_llm)

        if 'route_suggestions' in alt_routes:
            for route in alt_routes['route_suggestions']:
                st.markdown(f"**From:** {route['congested_street']}  →  **To:** {route['alternative_route']}  ")
                st.markdown(f"- Current Traffic: {route['current_traffic']}  ")
                st.markdown(f"- Estimated Time Saved: {route['estimated_time_saved']}  ")
                st.markdown("---")
        else:
            st.info("No alternative route suggestions available.")

#    Tab 5: Dynamic Toll Adjustment
    with tab5:
        st.markdown("<h3>Dynamic Toll Pricing Agent</h3>", unsafe_allow_html=True)

        with st.spinner("Adjusting tolls based on traffic..."):
            toll_info = adjust_dynamic_tolls(filtered_df, selected_street, use_llm)

        if 'toll_adjustments' in toll_info:
            for toll in toll_info['toll_adjustments']:
                st.markdown(f"**Route:** {toll['route']}  ")
                st.markdown(f"- Base Toll: ${toll['base_toll']}  ")
                st.markdown(f"- Adjusted Toll: ${toll['adjusted_toll']}  ")
                st.markdown(f"- Reason: {toll['reason']}  ")
                st.markdown("---")
        else:
            st.info("No toll adjustment data available.")

    # Tab 6: Traffic Statistics
    with tab6:
        st.markdown("<h3>Traffic Statistics</h3>", unsafe_allow_html=True)
        st.plotly_chart(show_hourly_distribution(filtered_df, selected_street), use_container_width=True)
        st.plotly_chart(show_vehicle_distribution(filtered_df, selected_street), use_container_width=True)

if __name__ == "__main__":
    main()
