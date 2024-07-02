import numpy as np

def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

# Ví dụ sử dụng
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
num_classes = 3  # Số lượng lớp
y_one_hot = to_one_hot(y, num_classes)
print(y_one_hot)