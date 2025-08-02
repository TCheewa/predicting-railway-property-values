import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
import os

# === Load dataset ===
file_path = "INPUT YOUR FILE PATH HERE"
df = pd.read_csv(file_path, low_memory=False)

# === Clean and standardize ===
df = df.dropna(subset=['Latitude', 'Longitude', 'station_lat', 'station_lon', 'real_price', 'date_of_transfer'])
df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer'], errors='coerce')
df = df[df['date_of_transfer'].notna()]
df = df[(df['date_of_transfer'].dt.year >= 2005) & (df['date_of_transfer'].dt.year <= 2025)]

# === Fix group names with hyphens ===
df['railway_period'] = df['railway_period'].astype(str).str.strip().str.replace("–", "-")

# === Sample records to reduce rendering load ===
df_sampled = df.groupby('town/city', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 2000), random_state=42))

# === Determine quartiles for price tiers ===
q1 = df['real_price'].quantile(0.25)
q2 = df['real_price'].quantile(0.50)
q3 = df['real_price'].quantile(0.75)

# === Define color and label mapping ===
def color_group(group):
    if group in ['Pre → Post', 'Pre', 'Post']:
        return 'blue'
    elif group == 'Control-Station':
        return 'green'
    elif group == 'Control-NoStation':
        return 'orange'
    return 'gray'

def label_background_color(group):
    if group in ['Pre → Post', 'Pre', 'Post']:
        return 'rgba(0, 102, 255, 0.8)'
    elif group == 'Control-Station':
        return 'rgba(0, 153, 0, 0.8)'
    elif group == 'Control-NoStation':
        return 'rgba(255, 153, 0, 0.8)'
    return 'rgba(120,120,120,0.6)'

# === Create geojson features ===
features = []
for _, row in df_sampled.iterrows():
    try:
        date_str = row['date_of_transfer'].date().isoformat()
    except:
        continue

    price = row['real_price']
    if price <= q1:
        size = 3
    elif price <= q2:
        size = 5
    elif price <= q3:
        size = 7
    else:
        size = 9

    color = color_group(row['railway_period'])

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [row['Longitude'], row['Latitude']],
        },
        "properties": {
            "time": date_str,
            "style": {"color": color},
            "icon": "circle",
            "iconstyle": {
                "fillColor": color,
                "fillOpacity": 0.6,
                "stroke": "true",
                "radius": size
            },
            "popup": f"""
                <b>Price:</b> £{int(price)}<br>
                <b>Town:</b> {row['town/city']}<br>
                <b>Group:</b> {row['railway_period']}<br>
                <b>Station:</b> {row['nearest_station_name']}<br>
                <b>Distance:</b> {row['distance_to_station'] / 1000:.2f} km<br>
                <b>Date:</b> {date_str}
            """
        }
    })

# === Create map ===
m = folium.Map(location=[52.5, -1.5], zoom_start=6)

# === Add time slider ===
TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="P1Y",
    add_last_point=True,
    auto_play=False,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY",
    time_slider_drag_update=True,
).add_to(m)

# === Add station markers ===
stations = df_sampled[['nearest_station_name', 'station_lat', 'station_lon']].drop_duplicates()
for _, row in stations.iterrows():
    folium.Marker(
        location=[row['station_lat'], row['station_lon']],
        icon=folium.Icon(color='red', icon='train', prefix='fa'),
        popup=f"🚉 {row['nearest_station_name']}"
    ).add_to(m)

# === Add town labels ===
town_centers = df_sampled.groupby('town/city')[['Latitude', 'Longitude']].median().reset_index()
town_groups = df_sampled.groupby('town/city')['railway_period'].first().reset_index()
town_centers = pd.merge(town_centers, town_groups, on='town/city', how='left')

for _, row in town_centers.iterrows():
    bg_color = label_background_color(row['railway_period'])
    label_html = f"""
    <div style="
        font-size: 14px;
        background-color: {bg_color};
        color: black;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: bold;
        border: 1px solid #444;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        display: inline-block;
        white-space: nowrap;
    ">
        {row['town/city'].title()}
    </div>
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(html=label_html)
    ).add_to(m)

# === Legend 1 ===
legend_html = """
<div style="position: fixed; top: 20px; right: 20px; z-index: 9999; background: white; padding: 10px; border:2px solid gray;">
<b>Legend (Group)</b><br>
<span style="color:blue;">●</span> Pre / Post<br>
<span style="color:green;">●</span> Control–Station<br>
<span style="color:orange;">●</span> Control–NoStation<br>
<span style="color:red;">📍</span> Nearest Station<br>
<em>Circle size = Price tier (4 groups)<br>Date = Year of transaction</em>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# === Legend 2 ===
legend2_html = """
<div style="position: fixed; top: 50%; right: 20px; transform: translateY(-50%); z-index: 9999; background: white; padding: 10px; border:2px solid gray; max-width: 180px; font-size: 14px;">
<b>Key Transition Years</b><br>
Bicester: 2024 (Planned)<br>
Corby: 2009<br>
Kenilworth: 2018<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend2_html))

# === Save HTML ===
output_file = "INPUT YOUR FILE PATH HERE"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
m.save(output_file)
print("Map created and saved as:", output_file)
