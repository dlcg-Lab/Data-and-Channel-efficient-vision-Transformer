import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib as mpl

config = {
    "font.family": 'serif',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}

plt.ylim([90.5, 94.25])
plt.xlim([15, 155])


class point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z * 200


plt.xlabel('Params', fontsize=15)
plt.ylabel('Top1 Accuracy [%]', fontsize=15)

DCT = point(28.96, 93.24, 5.72)
VGG19 = point(139.61, 92.27, 19.64)
ResNet50 = point(23.58, 91.13, 4.11)
ResNet101 = point(42.62, 93.89, 7.83)
ViT = point(28.95, 91.67, 5.69)

cmap = mpl.cm.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, cmap.N))

plt.scatter(DCT.x,  # 横坐标
            DCT.y,  # 纵坐标
            label='1',
            s=DCT.z)  # 标签 即为点代表的意思

plt.scatter(VGG19.x,  # 横坐标
            VGG19.y,  # 纵坐标
            label='2',
            s=VGG19.z)  # 标签 即为点代表的意思

plt.scatter(ResNet50.x,  # 横坐标
            ResNet50.y,  # 纵坐标
            label='3',
            s=ResNet50.z)  # 标签 即为点代表的意思

plt.scatter(ResNet101.x,  # 横坐标
            ResNet101.y,  # 纵坐标
            label='4',
            s=ResNet101.z)  # 标签 即为点代表的意思

plt.scatter(ViT.x,  # 横坐标
            ViT.y,  # 纵坐标
            label='5',
            s=ViT.z)  # 标签 即为点代表的意思

plt.legend('best', labels=['DCT', 'VGG19', 'ResNet50', 'ResNet101', 'ViT'], markerscale=0.2)
plt.savefig('out3.png', dpi=100)
plt.show()
