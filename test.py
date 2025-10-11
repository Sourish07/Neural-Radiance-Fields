import tempfile

import lovely_tensors as lt

lt.monkey_patch()

with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
    file_name = tmp_file.name
    print(file_name)

from nerf_dataset import TinyCybertruckDataset

test_data = TinyCybertruckDataset(split="test")
print(test_data[:])
