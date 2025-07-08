from gents.dataset.base import WebDownloadDataModule
# from src.dataset.base import BaseDataModule


class ETTh1(WebDownloadDataModule):
    D = 7
    index_col = 'date'
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh1.csv"
    data_source = 'csv'
    
    @property
    def dataset_name(self) -> str:
        return "ETTh1"

class ETTh2(ETTh1):
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh2.csv"

    @property
    def dataset_name(self) -> str:
        return "ETTh2"

class ETTm1(ETTh1):
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm1.csv"
    
    
    @property
    def dataset_name(self) -> str:
        return "ETTm1"

class ETTm2(ETTh1):
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv"
    
    @property
    def dataset_name(self) -> str:
        return "ETTm2"
