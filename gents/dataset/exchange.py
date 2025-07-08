from gents.dataset.base import WebDownloadDataModule
# from src.dataset.base import BaseDataModule


class Exchange(WebDownloadDataModule):
    D = 8
    index_col = 'date'
    url = "https://drive.google.com/file/d/1rN79CxW3Vldp-WDuSoG0bKq9tYQR79UK/view?usp=share_link"
    csv_dir = 'exchange_rate/exchange_rate.csv'
    data_source = 'zip'
    
    @property
    def dataset_name(self) -> str:
        return "Exchange_rate"

# class Exchange(BaseDataModule):
#     D = 8
#     url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz"
    
#     def __init__(
#         self,
#         seq_len: int = 24,
#         select_seq_dim: List[int | str] = None,
#         batch_size: int = 32,
#         data_dir: Path | str = Path.cwd() / "data",
#         condition: str = None,
#         scale: bool = True,
#         inference_batch_size: int = 1024,
#         max_time: float = None,
#         add_coeffs: str = None,
#         **kwargs,
#     ):
#         super().__init__(
#             seq_len,
#             len(select_seq_dim) if isinstance(select_seq_dim, list) else self.D,
#             condition,
#             batch_size,
#             inference_batch_size,
#             max_time,
#             add_coeffs,
#             data_dir,
#             **kwargs,
#         )
#         self.scale = scale
#         self.select_seq_dim = select_seq_dim
#         if select_seq_dim is not None:
#             assert max(select_seq_dim) < self.D

#     def get_data(self):
#         pre_download_dir = self.data_dir / f"{self.dataset_name}_raw.csv"
#         if pre_download_dir.exists():
#             # Load data from local file if it exists
#             df = pd.read_csv(pre_download_dir)
#         else:
#             # Download stock data
#             headers = {"Authorization": "Test"}

#             response = requests.get(self.url, headers=headers)
#             with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
#                 decompressed_content = gz_file.read().decode('utf-8')  # 二进制转字符串
#                 df = pd.read_csv(StringIO(decompressed_content), header=None)
#                 df.to_csv(pre_download_dir, index=False)

#         # select dimensions
#         if self.select_seq_dim is not None:
#             if isinstance(self.select_seq_dim[0], str):
#                 df = df[self.select_seq_dim]
#             elif isinstance(self.select_seq_dim[0], int):
#                 df = df.iloc[:, self.select_seq_dim]
                
#         data_raw = df.values.astype(np.float32)
#         n_window = data_raw.shape[0] - self.total_seq_len + 1
#         n_trainval_window = int(n_window * 0.7) + int(n_window * 0.2)
#         n_trainval_timesteps = n_trainval_window + self.total_seq_len - 1

#         # scale
#         if self.scale:
#             self.scaler = StandardScaler()
#             self.scaler.fit(data_raw[:n_trainval_timesteps])
#             data_raw = self.scaler.transform(data_raw)
#         data_raw = torch.from_numpy(data_raw)

#         # slide window
#         data = []
#         for i in range(len(data_raw) - self.total_seq_len + 1):
#             data.append(data_raw[i : i + self.total_seq_len, :])
#         data = torch.stack(data, dim=0).float()
#         data_mask = torch.isnan(data)
#         class_label = None

#         return data, data_mask, class_label

#     @property
#     def dataset_name(self) -> str:
#         return "Exchange_rate"
