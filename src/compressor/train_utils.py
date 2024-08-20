import numpy as np
from typing import Optional, Tuple, List


# # 示例数组
# my_array = np.array([1, 2, 3, 4, 5, 6])
# # 要删除的元素索引
# indices_to_remove = [1, 3, 5]
# # 删除元素
# filtered_array = remove_elements_by_indices_np(my_array, indices_to_remove)
# print(filtered_array)  # 输出: [1 3 5]
def remove_elements_by_indices_np(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    mask = np.ones(len(arr), dtype=bool)
    mask[indices] = False
    return arr[mask]
