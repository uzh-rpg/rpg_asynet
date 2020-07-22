import torch
from enum import Enum


class Sites(Enum):
    INACTIVE_SITE = 0
    ACTIVE_SITE = 1
    UPDATED_SITE = 2
    NEW_ACTIVE_SITE = 3
    NEW_INACTIVE_SITE = 4
    VISUALIZATION_UPDATE_LOCATION = 5
