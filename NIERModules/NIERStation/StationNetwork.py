from typing import List, Optional, Tuple
import pandas as pd
import json
import pickle


class StationNetwork:
    DEFAULT_STATION_GROUPS_PATH = '/home/0_code/OpenWebUI/pipelines/NIERModules/NIERStation/similarity_results_v2.pkl'
    
    try:
        with open(DEFAULT_STATION_GROUPS_PATH, "rb") as f:
            DEFAULT_STATION_GROUPS = pickle.load(f)
            
    except FileNotFoundError:
        print(f"[Error] Pickle file '{DEFAULT_STATION_GROUPS_PATH}' not found.")
        DEFAULT_STATION_GROUPS = {}
        

    def __init__(self,
                 station_groups: Optional[List[List[int]]] = None,
                 similarity_pickle_path: Optional[str] = None):
        """
        Initialize with a list of station groups.
        If no station groups are provided, default groups are used.
        Args:
            station_groups (Optional[List[List[int]]]): User Defined Station Groups.
            similarity_pickle_path (Optional[str]): Pickle file path containing Similarity between stations.
        """
        if station_groups is None:
            station_groups = self.DEFAULT_STATION_GROUPS


    def get_station_list(self) -> List[int]:
        """
        Returns the list of all station IDs in the network.
        """
        return list(self.DEFAULT_STATION_GROUPS.keys())
    
    
    def get_related_station(self, station_id: int, element: 'str') -> List[int]:
        """
        Returns the group of related stations for a given station ID,
        excluding the station itself.
        """
        if self.station_exists(station_id):
            related_stations = list(self.DEFAULT_STATION_GROUPS[station_id][element].keys())
            if related_stations is not None:
                return related_stations
            else:
                print(f"[Error] No related stations found for station ID {station_id}.")
                return []
        else:
            print(f"[Error] Station ID {station_id} not found.")
            return []
        
        
    def station_exists(self, station_id: int) -> bool:
        """
        Checks if a station ID exists in the network.
        """
        return station_id in self.DEFAULT_STATION_GROUPS


    def search_similarity(self, station_a: int, station_b: int, element: str, window_size: str) -> tuple:
        """
        Searches for the similarity between two stations for a given element and window size.

        Args:
            station1 (str): The first station ID.
            station2 (str): The second station ID.
            element (str): The pollutant element (e.g., "NO2", "PM10").
            window_size (str): The similarity window size (e.g., "6h", "24h").

        Returns:
            float, float: The DTW distance and similarity score.
        """
        if self.DEFAULT_STATION_GROUPS is None:
            raise ValueError("Similarity data is not loaded.")
        
        sim_data = self.DEFAULT_STATION_GROUPS[station_a][element][station_b]
        
        if sim_data is None:
            raise ValueError(f"No similarity data found for stations {station_a} and {station_b}.")

        # Extract distance and standard deviation for the given window size
        
        avg_dist_key = f"avg_dist_{window_size}"
        sd_key = f"sd_{window_size}"
        
        if avg_dist_key not in sim_data or sd_key not in sim_data:
            raise ValueError(f"No data found for window size {window_size} in element {element}.")

        distance = sim_data[avg_dist_key]
        standard_deviation = sim_data[sd_key]

        return distance, standard_deviation
