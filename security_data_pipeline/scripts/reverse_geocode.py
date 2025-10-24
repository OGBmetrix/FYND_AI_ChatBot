from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import pandas as pd

# Load your parquet
df = pd.read_parquet("../data/security_master_geo.parquet")

# Initialize geocoder (use user_agent for Nominatim)
geolocator = Nominatim(user_agent="FYND_AI_GeoTagger")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# Create a function to get location names
def get_location(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    try:
        location = reverse((lat, lon), language="en", exactly_one=True)
        if location and "address" in location.raw:
            addr = location.raw["address"]
            # Try to extract neighborhood / city / suburb
            return (
                addr.get("neighbourhood")
                or addr.get("suburb")
                or addr.get("city")
                or addr.get("town")
                or addr.get("county")
            )
        return None
    except Exception:
        return None

# Only process rows missing a name
mask = df["location_name"].isna() | (df["location_name"] == "") | (df["location_name"] == "Unknown")
subset = df.loc[mask, ["latitude", "longitude"]].dropna()

tqdm.pandas(desc="Reverse geocoding")
df.loc[mask, "location_name"] = subset.progress_apply(lambda x: get_location(x["latitude"], x["longitude"]), axis=1)

# Save enriched parquet
df.to_parquet("../data/security_master_geo_enriched.parquet", index=False)
print("✅ Enriched file saved as security_master_geo_enriched.parquet")