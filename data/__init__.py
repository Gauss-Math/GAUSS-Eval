from .IMO2025.imo import IMO2025Dataset
from .USAMO2025.usamo import USAMO2025Dataset
from .DEBUG.debug import DEBUGDataset

get_dataset_fn = {
    "IMO2025": IMO2025Dataset(),
    "USAMO2025": USAMO2025Dataset(load_from_old=True),
    "DEBUG": DEBUGDataset()
}