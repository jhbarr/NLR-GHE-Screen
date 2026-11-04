import pandas as pd
from pathlib import Path

from building_energy_profiles.composite import pull_sized_time_series

# ---------------------------------------------------------------------------
# Renaming conventions for Comstock / Restock data columns
# ---------------------------------------------------------------------------

COMSTOCK_RENAME = {
    'out.district_cooling.cooling.energy_consumption': 'cooling_load',
    'out.district_heating.heating.energy_consumption': 'heating_load'
}

RESTOCK_RENAME = {
    'out.load.cooling.energy_delivered': 'cooling_load',
    'out.load.heating.energy_delivered': 'heating_load',
}

# ---------------------------------------------------------------------------
# Building load profile aggregation
# ---------------------------------------------------------------------------

def get_building_load_profiles(
        building_types: list[str],
        building_sizes: list[int],
        building_sources: list[str],
        state: str,
        counties: str | list[str] = 'All'
):
    """Downloads and returns representative load profiles for the specified buildings

    This function is used to aggregate different load profile data about various Comstock / Restock 
    building types scaled to the specified sizes. For space usage purposes, the raw metadata used by the 
    internal functions are temporary and only the load profile data remains. Therefore, recalculations redownload
    data.

    Args:
        buildings_types (list[str]): A list of the building types whose load profiles should be aggregated
            These must match either Comstock or Restock naming conventions
        building_sizes (list[int]): The square footage of each building 
        building_sources (list[str]): Whether a building type is in the comstock or restock database
        state (str): The US state abbreviation fot the state the building data should come from
        counties (str | list[str]): A name or list of names of the counties within the state the building data should come from
            Defaults to 'All'
    
    Returns:
        list[pd.DataFrame]: A list of dataframes with the net load curves of each building 
    """
    # Create the directories for the metadata and timeseries data
    save_dir, metadata_dir = Path('./datasets'), Path('./metadata')
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for type, size, source in zip(building_types, building_sizes, building_sources):
        try:
            df = pull_sized_time_series(
                building_type=type,
                sqft=size,
                state=state,
                county_name=counties,
                save_dir=save_dir,
                metadata_dir=metadata_dir,
                ideal_thermal_air_loads=True,
                sqft_tolerance_pct=40.0,
                cache_metadata=False,
                product=source
            )
        except (ValueError) as e:
            print(f"Error: {e}")
            continue

        df = df.set_index('timestamp')
        if source == 'restock':
            df = df.rename(columns=RESTOCK_RENAME)
        else:
            df = df.rename(columns=COMSTOCK_RENAME)

        heating_cop = 3
        cooling_cop = 4.5

        df['net_load'] = (df['heating_load'] - (df['heating_load'] / heating_cop)) - (df['cooling_load'] + (df['cooling_load'] / cooling_cop))
        # df['net_heating'] = df['net_load'].clip(lower=0)
        # df['net_cooling'] = (-1 * df['net_load']).clip(lower=0)

        df.attrs['bldg_type'] = type
        frames.append(df[['net_load', 'heating_load', 'cooling_load']])

    return frames