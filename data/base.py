from typing_extensions import override
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
class BaseDataset:
    def __init__(self, data_name: str = "BaseDataset", data_path: str = None):
        self.data_name = data_name
        self.data_path = data_path
        self.data: List[Dict[str, Any]] = []
    
    @override
    def __str__(self):
        """
        print the dataset information
        """
        return f"Dataset: {self.data_name}"
    @override
    def load_data(self):
        """
        load a json format data
        """
        raise NotImplementedError

    @override   
    def parse_prompt(self, data_item) -> dict:
        """
        parse the prompt from the data item

        Args:
            data_item: a dictionary of data item

        Returns:
            dict: a dictionary of parsed prompt
        """
        raise NotImplementedError

    @override
    def parse_response(self, response, data_item):
        """
        parse the response from the data item

        Args:
            response: a dictionary of response
            data_item: a dictionary of data item

        Returns:
            dict: a dictionary of parsed response
        """
        raise NotImplementedError


class AbstractResultEntry(ABC):
    """Abstract base class defining the interface for result entries."""
    
    @classmethod
    @abstractmethod
    def from_json_data(cls, data: Dict[str, Any], source_file: str = "") -> 'AbstractResultEntry':
        """Create a ResultEntry from JSON data."""
        pass


class AbstractResultSet(ABC):
    """Abstract base class defining the interface for result sets."""
    pass