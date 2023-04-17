import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

table = np.loadtxt('result2.csv', delimiter=',')

epoch = table[:, 0]
DCT = table[:, 1]
DCT_E = table[:, 2]
DCT_N = table[:, 3]

plt.figure(figsize=(12, 10))
plt.axis('off')
plt.subplot(2, 1, 1)
plt.title('Global', fontsize=15)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Top1 Accuracy [%]', fontsize=20)
plt.plot(epoch, DCT, label='ResNet50', linewidth=2)
plt.plot(epoch, DCT_E, label='ResNet50-E', linewidth=2)
plt.plot(epoch, DCT_N, label='ResNet50-N', linewidth=2)
plt.legend(fontsize=20, loc='lower right')

plt.subplot(2, 1, 2)
plt.title('Partial', fontsize=15)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Top1 Accuracy [%]', fontsize=20)
plt.plot(epoch[-3:], DCT[-3:], label='ResNet50', linewidth=2)
plt.plot(epoch[-3:], DCT_E[-3:], label='ResNet50-E', linewidth=2)
plt.plot(epoch[-3:], DCT_N[-3:], label='ResNet50-N', linewidth=2)
plt.legend(fontsize=20, loc='lower right')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.suptitle('ResNet Group', fontsize=20)
plt.savefig('out2.png', dpi=100)
plt.show()

