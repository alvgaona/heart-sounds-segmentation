import os
from typing import Any
import pandas as pd
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from torch.hub import download_url_to_file
from torchaudio.datasets.utils import _extract_zip as extract_zip

from hss.utils.files import walk_files

class PhysionetChallenge2016(Dataset):
    """ Heart Sounds Dataset from PhysioNet Challenge 2016 """
    _ext_audio = ".wav"
    _ext_reference = ".csv"

    def __init__(
        self,
        root: str,
        url: str = "training",
        train: bool = True,
        folder_in_archive: str = "training",
        download: bool = False,
        transform = None,
        ):
        """
        Instantiate HeartSoundsAudio object.

        Args:
            root_dir (str): Directory with all the images.
            train (bool, optional): If True, creates dataset from training,
            otherwise from test.pt
            download (bool):
        """
        self._root = root
        self._train = train
        self._transform = transform

        if train is False:
            url = "validation"

        ext_archive = ".zip"
        base_url = "https://www.physionet.org/files/challenge-2016/1.0.0/"
        url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]

        folder_in_archive = basename
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url_to_file(url + "?download", f"{root}/{archive}")
                    extract_zip(archive, to_path=f"{root}/{basename}")

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=True, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        file_id = self._walker[n]
        reference = "REFERENCE"

        path = os.path.dirname(file_id)
        df = pd.read_csv(
          os.path.join(path, reference + self._ext_reference),
          names=["ID", "Condition"]
        )

        set_name = path.split("/")[-1]
        basename = os.path.basename(file_id)

        record = df.loc[df["ID"].isin([basename])]
        label = record.iloc[0]["Condition"]

        file_audio = file_id + self._ext_audio
        output, sample_rate = torchaudio.load(file_audio)

        output = self._transform(output[0])

        return (
            output,
            sample_rate,
            label,
            set_name,
            basename
        )

    def __len__(self):
        return len(self._walker)

    def collate_fn(self, batch):
        batch_size = len(batch)

        if (batch_size == 1):
            return batch

        tensors = [torch.reshape(t[0], (-1, 1)) for t in batch]
        padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors)

        for i in range(0, batch_size):
            batch[i] = list(batch[i])
            batch[i][0] = torch.reshape(padded_tensors[:, i], (1, -1))
            batch[i] = tuple(batch[i])

        return batch
    

class DavidSpringerHSS(Dataset):
    def __init__(self, root: str) -> None:
        self.root = root
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass
