#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_preqnt_dataset(data_file: str, split: str):
    splits = ["train", "test", "val"]
    if split not in splits:
        print(f"Invalid dataset split: {split}")
        assert False
    
    if split == "train":
        # Load train data
        data = np.load(data_file + "/train_data.npz")
    elif split == "test":
        # Load test data
        data = np.load(data_file + "/test_data.npz")
    elif split == "val":
        # Load validation data
        data = np.load(data_file + "/val_data.npz")

    X = torch.from_numpy(data["features"]).float()
    y = torch.from_numpy(data["labels"]).float()

    return TensorDataset(X, y)





