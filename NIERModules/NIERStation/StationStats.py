import pandas as pd
import pickle
from typing import Optional, Union

class StationStats:
    def __init__(self, StationStatDict: Optional[Union[dict, str]] = None):
        """
        Initialize with a Station Stats Dictionary.

        Args:
            StationStatDict (Optional[Union[dict, str]], optional): 
                User Defined Station Stats Dictionary as a dict or path to a pickle file. Defaults to None.
        """        
        
        self.DECAULT_STATION_STATS_PATH = "/home/0_code/NIER_Pipelines/NIERModules/NIERStation/monthly_stats_v1.pkl"
        self.station_stats_dict = self._load_station_stats_dict(StationStatDict)
        
    def _load_station_stats_dict(self, StationStatDict: Optional[Union[dict, str]] = None) -> dict:
        """
        Loads the Station Stats Dictionary from a given Dictionary or a pickle file.

        Args:
            StationStatDict (Optional[Union[dict, str]], optional): 
                User Defined Station Stats Dictionary as a dict or path to a pickle file. Defaults to None.

        Returns:
            dict: Station Stats Dictionary
        """        
        
        if isinstance(StationStatDict, dict):
            return StationStatDict
        
        if isinstance(StationStatDict, str):
            try:
                with open(StationStatDict, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                print(f"[Error] Pickle file '{StationStatDict}' not found. Using default network.")
                
                
        try: 
            with open(self.DECAULT_STATION_STATS_PATH, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"[Error] Default pickle file '{self.DECAULT_STATION_STATS_PATH}' not found. Returning empty DataFrame.")
            return {}  # Return empty Dictionary as fallback
    def station_exists(self, station_id: int) -> bool:
        """
        Checks if a station ID exists in the network.

        Args:
            station_id (int): Station ID

        Returns:
            bool: True if the station exists, False otherwise.
        """        
        return station_id in self.station_stats_dict.keys()
        

    def get_station_stats_dataframe(self, station_id: int) -> Optional[pd.DataFrame]:
        """
        Returns the stats of a given station ID.

        Args:
            station_id (int): Station ID

        Returns:
            Optional[dict]: Station Stats Dictionary
        """        
        if not self.station_exists(station_id):
            print(f"[Error] Station ID {station_id} not found.")
            return None
        
        return self._load_station_stats_dict().get(station_id)
    
    def get_avg_std(self, station_id: int, element: str, month: int) -> Optional[tuple]:
        """
        Returns the average and standard deviation of a given element for a given month.

        Args:
            station_id (int): Station ID
            element (str): Element to calculate the average and standard deviation
            month (int): Month to calculate the average and standard deviation

        Returns:
            tuple: (Average, Standard Deviation)
        """        
        if not self.station_exists(station_id):
            print(f"[Error] Station ID {station_id} not found.")
            return None

        station_stats = self.get_station_stats_dataframe(station_id)
        if station_stats is None:
            print(f"[Error] Station Stats not found for station ID {station_id}.")
            return None

        # YearMonth 컬럼이 문자열이면 변환 후 월(month)과 비교
        if isinstance(station_stats.index, pd.Index):  
            station_stats = station_stats.reset_index()  # Index를 컬럼으로 변환

        station_stats["Month"] = station_stats["YearMonth"].astype(str).str[-2:].astype(int)

        
        # 지정된 month에 해당하는 데이터만 필터링
        # Note: 9999-12는 전체 기간의 평균, 표준편차임.
        monthly_stats = station_stats[station_stats['YearMonth'].notna() & 
                                     (station_stats['YearMonth'].astype(str).str[-2:].astype(int) == month) & 
                                     (station_stats['YearMonth'] != '9999-12')]

        if monthly_stats.empty:
            print(f"[Warning] No data available for station ID {station_id} in month {month}.")
            return None

        element = element.upper()
        mean_col = f"{element}_mean"
        std_col = f"{element}_std"
        
        if mean_col not in monthly_stats.columns or std_col not in monthly_stats.columns:
            print(f"[Error] Element '{element}' not found in station data.")
            return None
        
        element_monthly_mean = monthly_stats[mean_col].mean()
        element_monthly_std = monthly_stats[std_col].mean()

        return (element_monthly_mean, element_monthly_std)