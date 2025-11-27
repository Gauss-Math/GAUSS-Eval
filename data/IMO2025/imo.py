from ..base import BaseDataset
import json
from typing import List, Dict, Any
from .eval_utils import parse_prompt
from datasets import load_dataset

class IMO2025Dataset(BaseDataset):
    """
    IMO2025 dataset
    """
    def __init__(self, data_name: str = "IMO2025", data_path: str = "data/IMO2025/datasource/matharena_imo2025.jsonl", load_from_old: bool = False):
        super().__init__(data_name)
        self.data: List[Dict[str, Any]] = []
        self.data_path = data_path
        self.load_data(load_from_old)

    def __str__(self):
        """
        print the dataset information
        """
        return f"IMO2025: {self.data_name}, length: {len(self.data)}, path: {self.data_path}"
    
    def load_data(self, load_from_old: bool = False):
        """
        load the data
        """
        if load_from_old:
            # load data from old dataset
            i = 0
            with open(self.data_path, 'r') as f:
                for line in f:
                    data_item = json.loads(line)
                    data_item['id'] = i
                    self.data.append(data_item)
                    i += 1
            print(f"Loaded {len(self.data)} old dataset entries")  
        else:
            # load jsonl data
            with open(self.data_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
            
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

