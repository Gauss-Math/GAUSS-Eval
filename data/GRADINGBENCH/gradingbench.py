from ..base import BaseDataset
import json
from typing import List, Dict, Any
from .eval_utils import parse_prompt
from datasets import load_dataset
import pandas as pd

class GRADING_BENCHDataset(BaseDataset):
    """
    GRADING_BENCH dataset
    """
    def __init__(self, data_name: str = "GRADING_BENCH", data_path: str = "data/GRADINGBENCH/datasource/gradingbench.csv", load_from_old: bool = False):
        super().__init__(data_name)
        self.data: List[Dict[str, Any]] = []
        self.data_path = data_path
        self.load_data(load_from_old)

    def __str__(self):
        """
        print the dataset information
        """
        return f"USAMO2025: {self.data_name}, length: {len(self.data)}, path: {self.data_path}"
    
    def load_data(self, load_from_old: bool = False):
        """
        load the data
        """
        # load data from csv
        self.data = pd.read_csv(self.data_path)
        # transform to list of dicts
        self.data = self.data.to_dict(orient='records')
        # name each item with the id
        for i, data_item in enumerate(self.data):
            data_item['id'] = i
    
    def parse_prompt(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        parse the prompt from the data item
        """
        return parse_prompt(data_item)

    
    def parse_response(self, response: Dict[str, Any], data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        parse the response from the data item
        """
        data_item['response'] = response['choices'][0]['message']['content']
        data_item['prompt'] = self.parse_prompt(data_item)
        return data_item

if __name__ == "__main__":
    dataset = GRADING_BENCHDataset()
    print(dataset.data.head())