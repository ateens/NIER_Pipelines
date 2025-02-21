import pandas as pd
import pickle
from typing import Optional, Union, List, Tuple
from haversine import haversine


class GeoSpatialNetwork:
    def __init__(self, geo_spatial_network: Optional[Union[pd.DataFrame, str]] = None):
        """
        Initialize with a Geo Spatial Network.
        If no network is provided, the default network is loaded from a pickle file.

        Args:
            geo_spatial_network (Optional[Union[pd.DataFrame, str]]): 
                User Defined Geo Spatial Network as a DataFrame or path to a pickle file.
        """
        self.DEFAULT_GEO_SPATIAL_NETWORK_PATH = "/home/1_Dataset/NIER/StationGeoData.pkl"
        self.geo_spatial_network = self._load_geo_spatial_network(geo_spatial_network)


    def _load_geo_spatial_network(self, geo_spatial_network: Optional[Union[pd.DataFrame, str]]) -> pd.DataFrame:
        """Loads the Geo Spatial Network from a given DataFrame or a pickle file."""
        if isinstance(geo_spatial_network, pd.DataFrame):
            return geo_spatial_network
        
        if isinstance(geo_spatial_network, str):  # If a file path is provided
            try:
                with open(geo_spatial_network, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                print(f"[Error] Pickle file '{geo_spatial_network}' not found. Using default network.")

        # Attempt to load the default pickle file
        try:
            with open(self.DEFAULT_GEO_SPATIAL_NETWORK_PATH, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"[Error] Default pickle file '{self.DEFAULT_GEO_SPATIAL_NETWORK_PATH}' not found. Returning empty DataFrame.")
            return pd.DataFrame()  # Return empty DataFrame as fallback

    
    def calculate_distances(self, station_a: int, station_b: int) -> Optional[float]:
        if not (self.station_exists(station_a) and self.station_exists(station_b)):
            print(f"[Error] Station ID {station_a} or {station_b} not found.")
            return None

        loc_a = self.get_lat_lon_by_station_id(station_a)
        loc_b = self.get_lat_lon_by_station_id(station_b)

        if loc_a is None or loc_b is None:
            print(f"[Error] Missing latitude/longitude for one of the stations: {station_a}, {station_b}.")
            return None

        return haversine(loc_a, loc_b, unit='km')


    def station_exists(self, station_id: int) -> bool:
        """
        Checks if a station ID exists in the network.
        
        Args:
            station_id (int): Station ID.
        
        Returns:
            bool: True if the station exists, False otherwise.
        """
        return station_id in self.geo_spatial_network['station_id'].values


    def get_near_stations_by_radius(self, station_id: int, radius: int) -> List[Tuple[int, float]]:
        """
        Get the list of stations within a given radius(KM) of a station, sorted by distance.

        Args:
            station_id (int): Station ID.
            radius (int): Radius in KM.

        Returns:
            List[Tuple[int, float]]: List of (station_id, distance) tuples within the given radius,
                                     sorted by closest distance first.
        """
        if not self.station_exists(station_id):
            print(f"[Error] Station ID {station_id} not found.")
            return []

        try:
            target_station = self.geo_spatial_network[self.geo_spatial_network['station_id'] == station_id]
            target_lat, target_lon = target_station[['latitude', 'longitude']].values[0]
        except IndexError:
            print(f"[Error] Missing latitude/longitude for Station ID {station_id}.")
            return []

        nearby_stations = []

        for _, row in self.geo_spatial_network.iterrows():
            if row['station_id'] == station_id:
                continue  # Skip the target station itself

            station_lat, station_lon = row['latitude'], row['longitude']
            distance = haversine((float(target_lat), float(target_lon)), 
                                 (float(station_lat), float(station_lon)), unit='km')

            if distance <= radius:
                nearby_stations.append((row['station_id'], distance))  # (측정소 ID, 거리)

        # 거리가 가까운 순으로 정렬
        nearby_stations.sort(key=lambda x: x[1])
        return nearby_stations


    def get_stations_by_administrative_division(self, division: str) -> list:
        """
        Get the list of stations within a given administrative division.

        Args:
            division (str): Administrative Division.
        
        Returns:
            pd.DataFrame: DataFrame containing stations in the specified division.
        """
        filtered_stations = self.geo_spatial_network[
        (self.geo_spatial_network['city_1'].str.contains(division, case=False, na=False)) |
        (self.geo_spatial_network['city_2'].str.contains(division, case=False, na=False))
        ]

        if filtered_stations.empty:
            print(f"[Warning] No stations found in division: {division}")
        
        return filtered_stations['station_id'].values.tolist()


    def get_lat_lon_by_station_id(self, station_id:int) -> Optional[Tuple[float, float]]:
        """
        Get the latitude and longitude of a station by its ID.
        
        Args:
            station_id (int): Station ID.
        
        Returns:
            Tuple[float, float]: Latitude and Longitude of the station.
        """
        try:
            station_data = self.geo_spatial_network[self.geo_spatial_network['station_id'] == station_id]
            return float(station_data['latitude'].values[0]), float(station_data['longitude'].values[0])
        except IndexError:
            print(f"[Error] Station ID {station_id} not found.")
            return None
    
    