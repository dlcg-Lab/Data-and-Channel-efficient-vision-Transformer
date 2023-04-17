"""
Edge_Blurs
"""
import paddle
import cv2
import numpy as np
import matplotlib.pyplot as plt


class EdgeBlurs(object):
    def __init__(self, p=0.5, kernel_size=6):
        self.p = p
        self.kernel_size = kernel_size

    def _blurs(self, masked_tensor, batch_size, chan, img_h, img_w, dtype):
        for i in range(batch_size):
            # 边缘检测
            edges = cv2.Canny(masked_tensor[i], 100, 200)
            # 创建一个与原始图像相同大小的全黑遮罩
            mask = np.zeros_like(masked_tensor[i])
            # 将边缘标记为1
            mask[edges != 0] = 1
            # 根据概率随机模糊边缘
            for row in range(img_h):
                for col in range(img_w):
                    if mask[row, col, 0] == 1 and np.random.uniform(0, 1) < self.p:
                        # 生成随机大小的模糊核
                        for channel in range(chan):
                            kernel_height = np.random.randint(1, self.kernel_size + 1)
                            kernel_width = np.random.randint(1, self.kernel_size + 1)
                            kernel = np.ones((kernel_height, kernel_width), np.float32) / (kernel_height * kernel_width)
                            # 获取需要模糊的区域
                            start_row = max(0, row - kernel_height // 2)
                            end_row = min(masked_tensor.shape[1], row + kernel_height // 2 + 1)
                            start_col = max(0, col - kernel_width // 2)
                            end_col = min(masked_tensor.shape[2], col + kernel_width // 2 + 1)
                            region = masked_tensor[i, start_row:end_row, start_col:end_col, channel]
                            # 将模糊后的值赋给需要模糊的区域
                            region[:] = cv2.filter2D(region, -1, kernel)
        return masked_tensor

    def __call__(self, inputs):
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, axis=0)
        batch_size = inputs.shape[0]
        img_h = inputs.shape[1]
        img_w = inputs.shape[2]
        chan = inputs.shape[3]
        masked_tensor = inputs.copy()
        output = self._blurs(masked_tensor, batch_size, chan, img_h, img_w, inputs.dtype)
        if batch_size == 1:
            output = np.squeeze(output, axis=0)
        return output

# def main():
#     eb = EdgeBlurs()
#     import PIL.Image as Image
#     import numpy as np
#     paddle.set_device('cpu')
#     img = np.asarray(Image.open(r"F:\数据集\garbage\garbage\1\img_332.jpg")).astype('uint8')
#     plt.imshow(img)
#     plt.show()
#     new_img = eb(img)
#     print(type(new_img))
#     print(new_img.shape)
#     for img in new_img:
#         plt.imshow(img)
#         plt.show()
#         plt.close()
#
#
# if __name__ == "__main__":
#     main()
