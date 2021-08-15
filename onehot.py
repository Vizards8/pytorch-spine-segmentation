import numpy as np
import torch


"""
提前squeeze，去除最后一个axis=1，只有在保存为.nii.gz时需要
"""

def mask2onehot(mask, class_list):
    """
    Converts a segmentation mask (BatchSize,1,H,W) to (BatchSize,K,H,W) where the second dim is a one
    hot encoding vector

    """
    mask = mask.squeeze(1)
    semantic_map = []
    for colour in class_list:
        equality = np.equal(mask, colour)
        # class_map = np.all(equality, axis=-1)
        semantic_map.append(equality)
    semantic_map = np.stack(semantic_map, axis=1).astype(np.float32)
    return semantic_map


def onehot2mask(mask):
    """
    Converts a mask (BatchSize,K,H,W,1) to (BatchSize,H,W,1)
    or (BatchSize,K,H,W) to (BatchSize,H,W)
    and expand axis
    """
    _mask = np.argmax(mask, axis=1).astype(np.uint8)
    _mask = _mask[:, np.newaxis, :]
    return _mask


# if __name__ == '__main__':
#     mask = torch.tensor(
#         [[1, 0, 0],
#          [0, 2, 0],
#          [0, 0, 3]]
#     )
#     mask = mask[np.newaxis, np.newaxis, :, :, np.newaxis]
#     one_hot = mask2onehot(mask, [0, 1, 2, 3])
#     mask = onehot2mask(one_hot)
#     print(one_hot.shape, one_hot)
#     print(mask.shape, mask)
