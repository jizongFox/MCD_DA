from MCD_DA.classification.datasets.base_data_loader import BaseDataLoader
from MCD_DA.classification.datasets.svhn import load_svhn

b = BaseDataLoader()

svhn = load_svhn()