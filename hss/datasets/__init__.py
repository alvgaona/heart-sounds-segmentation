import logging

from hss.datasets.circor import CirCorDataset
from hss.datasets.physionet_2016 import PhysionetChallenge2016
from hss.datasets.springer import DavidSpringerHSS

log = logging.getLogger("datasets")

__all__ = ["CirCorDataset", "DavidSpringerHSS", "PhysionetChallenge2016"]
