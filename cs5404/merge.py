import sys
import torch
import os
import numpy as np

def main():

    in1 = sys.argv[1]
    in2 = sys.argv[2]
    out = sys.argv[3]
    merge(in1, in2, out)


def merge(in1, in2, out):
    path_3dpw = in1
    path_hm36 = in2
    out_path = out

    if os.path.exists(out_path):
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    data1 = torch.load(path_3dpw, map_location=torch.device(device))
    data2 = torch.load(path_hm36, map_location=torch.device(device))

    merged = {}

    for key in data1:
        if key not in data2:
            raise ValueError(f"Key {key} missing in second file")

        v1 = data1[key]
        v2 = data2[key]

        if torch.is_tensor(v1):
            merged[key] = torch.cat([v1, v2], dim=0)

        elif isinstance(v1, list):
            merged[key] = v1 + v2

        else:
            try:
                merged[key] = np.concatenate([v1, v2], axis=0)
            except:
                raise TypeError(f"Unsupported type for key {key}: {type(v1)}")

    torch.save(merged, out_path)

if __name__ == "__main__":
    main()
