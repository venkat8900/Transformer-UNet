import os
from pathlib import Path

p = './data/RTS_binary/train/instrument_dataset_1/images/frame000.jpg'
d = Path(p).resolve().parents[1]

c = os.path.basename(p)
print(c)
print(d)
task = 'binary_masks'
e = os.path.join(d, task, c)

print(e)