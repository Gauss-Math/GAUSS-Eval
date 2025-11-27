from ..base import BaseDataset
import json
from typing import List, Dict, Any
from .eval_utils import parse_prompt

class DEBUGDataset(BaseDataset):
    """
    DEBUG data. debug.jsonl is the same as matharena_imo2025.jsonl, but only contains the first 10 items.
    """
    def __init__(self, data_name: str = "DEBUG", data_path: str = "data/DEBUG/debug.jsonl"):
        super().__init__(data_name)
        self.data: List[Dict[str, Any]] = []
        self.data_path = data_path
        self.load_data()

    def __str__(self):
        """
        print the dataset information
        """
        return f"Dataset: {self.data_name}, length: {len(self.data)}"
    
    def load_data(self):
        """
        load the data
        """
        # load jsonl data
        with open(self.data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        # only keep the first 10 items
        self.data = self.data[:10]
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

        return data_item

