import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import re

df_path_raw = gpd.read_file("data/N06_HighwaySection_fixed.geojson", crs="EPSG:4326")
df_point_raw = gpd.read_file("data/N06_Joint_fixed.geojson", crs="EPSG:4326")

df_erratum = pd.read_csv("erratum.csv")

# Apply errata to df_path_raw (HighwaySection)
df_path_erratum = df_erratum[df_erratum["dataset"] == "HighwaySection"].copy()

# Function to apply errata to HighwaySection geometries
def apply_highway_section_erratum(df_path_raw, df_path_erratum):
    for index, row in df_path_erratum.iterrows():
        feature_id = row["feature_id"]
        description = row["description"]

        if "追加" in description:
            # Regex to extract two existing points and the new point
            match = re.search(
                r'\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]\s*,\s*\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]\s*の間に\s*.*?\s*\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]\s*を追加',
                description
            )
            if match:
                lon1, lat1 = float(match.group(1)), float(match.group(2))
                lon2, lat2 = float(match.group(3)), float(match.group(4))
                new_lon, new_lat = float(match.group(5)), float(match.group(6))
                
                p1_desc = (lon1, lat1)
                p2_desc = (lon2, lat2)
                new_p = (new_lon, new_lat)

                idx = df_path_raw[df_path_raw["N06_004"] == feature_id].index
                if not idx.empty:
                    geom = df_path_raw.loc[idx[0], "geometry"]
                    
                    if geom.geom_type == "LineString":
                        coords = list(geom.coords)
                        try:
                            idx1 = coords.index(p1_desc)
                            idx2 = coords.index(p2_desc)
                            if abs(idx1 - idx2) == 1:
                                coords.insert(max(idx1, idx2), new_p)
                                df_path_raw.loc[idx[0], "geometry"] = LineString(coords)
                                print(f"Inserted point into LineString {feature_id}")
                            else:
                                print(f"Warning: Points for {feature_id} are not consecutive in LineString.")
                        except ValueError:
                            print(f"Warning: Could not find points for {feature_id} in LineString.")
                            
                    elif geom.geom_type == "MultiLineString":
                        new_lines = []
                        found = False
                        for line in geom.geoms:
                            coords = list(line.coords)
                            if not found:
                                try:
                                    idx1 = coords.index(p1_desc)
                                    idx2 = coords.index(p2_desc)
                                    if abs(idx1 - idx2) == 1:
                                        coords.insert(max(idx1, idx2), new_p)
                                        found = True
                                        print(f"Inserted point into MultiLineString {feature_id}")
                                except ValueError:
                                    pass
                            new_lines.append(LineString(coords))
                        
                        if found:
                            from shapely.geometry import MultiLineString
                            df_path_raw.loc[idx[0], "geometry"] = MultiLineString(new_lines)
                        else:
                            print(f"Warning: Could not find consecutive points for {feature_id} in any part of MultiLineString.")
                    else:
                        print(f"Warning: Unsupported geometry type {geom.geom_type} for {feature_id}")
                else:
                    print(f"Warning: feature_id {feature_id} not found in df_path_raw.")
            else:
                print(f"Warning: Could not parse all coordinates from description: {description}")
    return df_path_raw

# Apply errata to df_point_raw (Joint)
df_point_erratum = df_erratum[df_erratum["dataset"] == "Joint"].copy()

# Function to apply errata to Joint geometries/attributes
def apply_joint_erratum(df_point_raw, df_point_erratum):
    for index, row in df_point_erratum.iterrows():
        feature_id = row["feature_id"]
        description = row["description"]

        idx = df_point_raw[df_point_raw["N06_015"] == feature_id].index # Use N06_015 for feature_id
        if not idx.empty:
            # Handle "修正" (correction) for Joint
            if "修正" in description:
                # Example: "三ケ日 を 三ヶ日 に修正"
                match = re.search(r'(.+) を (.+) に修正', description)
                if match:
                    old_name = match.group(1)
                    new_name = match.group(2)
                    # Update the 'N06_018' column which holds the name
                    df_point_raw.loc[idx[0], 'N06_018'] = new_name
                    print(f"Updated feature_id {feature_id}: changed name from '{old_name}' to '{new_name}'")
                else:
                    print(f"Warning: Could not parse correction from description: {description}")

            # Handle "追加" (add) for Joint (e.g., adding a new point feature)
            elif "追加" in description:
                # Example: "八戸JCT [ 141.458046, 40.4599 ] を追加"
                match = re.search(r'(.+?)\s*\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\] を追加', description)
                if match:
                    if len(match.groups()) == 3: # Expecting 3 groups: name, lon, lat
                        new_joint_name = match.group(1).strip()
                        lon = float(match.group(2))
                        lat = float(match.group(3))
                    else:
                        print(f"Warning: Regex matched but did not capture expected groups for Joint addition: {description}")
                        continue # Skip to next iteration if groups are not as expected
                    new_point_geom = Point(lon, lat)

                    # Create a new GeoDataFrame row and concatenate.
                    # Ensure all relevant columns are populated.
                    # Assuming N06_012, N06_013, N06_014, N06_019 can be default or null for new entries
                    new_row_data = {
                        'N06_012': pd.NA, # Default or parse if available
                        'N06_013': pd.NA,
                        'N06_014': pd.NA,
                        'N06_015': feature_id, # Use the feature_id from erratum.csv
                        'N06_016': None,
                        'N06_017': None,
                        'N06_018': new_joint_name,
                        'N06_019': None,
                        'geometry': new_point_geom
                    }
                    # Convert to GeoDataFrame for concatenation
                    new_gdf_row = gpd.GeoDataFrame([new_row_data], crs=df_point_raw.crs)
                    df_point_raw = pd.concat([df_point_raw, new_gdf_row], ignore_index=True)
                    # Convert specified columns to nullable integer type
                    for col in ['N06_012', 'N06_013', 'N06_014']:
                        if col in df_point_raw.columns:
                            df_point_raw[col] = df_point_raw[col].astype('Int64')
                    print(f"Added new Joint: {new_joint_name} (feature_id: {feature_id}) at {lon}, {lat}")
                else:
                    print(f"Warning: Could not parse new point coordinates for Joint from description: {description}")

            # Handle "削除" (delete) for Joint
            elif "削除" in description:
                # This means deleting the row with the given feature_id.
                df_point_raw = df_point_raw.drop(idx).reset_index(drop=True) # Reset index after drop
                print(f"Deleted Joint with feature_id: {feature_id}")
            else:
                print(f"Warning: Unhandled erratum type for Joint: {description}")
        else:
            print(f"Warning: feature_id {feature_id} not found in df_point_raw for Joint erratum.")
    return df_point_raw

# Apply the errata
df_path = apply_highway_section_erratum(df_path_raw, df_path_erratum)
df_point = apply_joint_erratum(df_point_raw, df_point_erratum)

# Save the modified dataframes
df_path.to_file("data/N06_HighwaySection_fixed.geojson", driver="GeoJSON")
df_point.to_file("data/N06_Joint_fixed.geojson", driver="GeoJSON")

print("Erratum application complete. Modified data saved to data/N06_HighwaySection_fixed.geojson and data/N06_Joint_fixed.geojson")
