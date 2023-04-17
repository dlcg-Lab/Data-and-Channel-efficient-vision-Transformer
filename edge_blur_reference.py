import cv2
import numpy as np
import matplotlib.pyplot as plt


def random_mask(image_tensor, p=0.5, kernel_size=2):
    """对图像张量进行随机边缘模糊

    Args:
        image_tensor: 图像张量，可以是三维（单张图像）或四维（批量图像）
        p: 遮挡概率，默认为0.5
        kernel_size: 模糊核大小，默认为3

    Returns:
        处理后的图像张量
    """
    masked_tensor = np.copy(image_tensor)
    if len(masked_tensor.shape) == 3:
        masked_tensor = np.expand_dims(masked_tensor, axis=0)
    batch_size = masked_tensor.shape[0]
    for i in range(batch_size):
        # 边缘检测
        print(masked_tensor[i].shape)
        print(type(masked_tensor))
        print(masked_tensor.dtype)
        print(masked_tensor.max())
        print(masked_tensor.min())
        edges = cv2.Canny(cv2.cvtColor(image_tensor[i], cv2.COLOR_RGB2BGR), 100, 200)
        # 创建一个与原始图像相同大小的全黑遮罩
        mask = np.zeros_like(image_tensor[i])
        # 将边缘标记为1
        mask[edges != 0] = 1
        # 根据概率随机模糊边缘

        for row in range(masked_tensor.shape[1]):
            for col in range(masked_tensor.shape[2]):
                if mask[row, col, 0] == 1 and np.random.uniform(0, 1) < p:
                        # 生成随机大小的模糊核
                    for channel in range(masked_tensor.shape[3]):
                        kernel_height = np.random.randint(1, kernel_size+1)
                        kernel_width = np.random.randint(1, kernel_size+1)
                        kernel = np.ones((kernel_height, kernel_width), np.float32) / (kernel_height * kernel_width)
                        # 获取需要模糊的区域
                        start_row = max(0, row - kernel_height // 2)
                        end_row = min(masked_tensor.shape[1], row + kernel_height // 2 + 1)
                        start_col = max(0, col - kernel_width // 2)
                        end_col = min(masked_tensor.shape[2], col + kernel_width // 2 + 1)
                        region = masked_tensor[i, start_row:end_row, start_col:end_col, channel]
                        # 将模糊后的值赋给需要模糊的区域
                        region[:] = cv2.filter2D(region, -1, kernel)
    if batch_size == 1:
        masked_tensor = np.squeeze(masked_tensor, axis=0)
    return masked_tensor

# Sample:

img = cv2.imread('me.jpg')

plt.imshow(img)
plt.show()

img_tensor = np.expand_dims(img, axis=0)

masked_img_tensor = random_mask(img_tensor, p=0.5)

masked_img = cv2.cvtColor(np.uint8(masked_img_tensor), cv2.COLOR_RGB2BGR)

print(masked_img_tensor)

plt.imshow(masked_img)
plt.show()
