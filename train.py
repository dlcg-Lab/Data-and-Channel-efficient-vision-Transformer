import paddle
import itertools
import log
import sys
import time
import datetime
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer.lr import LinearWarmup
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from Edge_blur import EdgeBlurs
import ViT.module


class Cosine(CosineAnnealingDecay):
    def __init__(self, lr, step_each_epoch, epochs, **kwargs):
        super(Cosine, self).__init__(learning_rate=lr, T_max=step_each_epoch * epochs)
        self.update_specified = False


class CosineWarmup(LinearWarmup):
    def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        assert epochs > warmup_epoch, "total epoch({}) should be larger than warmup_epoch({}) in CosineWarmup.".format(
            epochs, warmup_epoch)
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = Cosine(lr, step_each_epoch, epochs - warmup_epoch)

        super(CosineWarmup, self).__init__(learning_rate=lr_sch, warmup_steps=warmup_step, start_lr=start_lr,
                                           end_lr=end_lr)
        self.update_specified = False


sys.stdout = log.Logger('./log/log{}.txt'.format(
    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))),
    mode='w', encoding='utf-8')

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print(f'训练开始，当前时间为{start_time}')

    print("=======begin=======")

    img_size = 128

    drop = 0.2

    net = ViT.module.VisionTransformer(
        img_size=img_size,
        patch_size=16,
        class_dim=10,
        embed_dim=384,
        depth=4,
        num_heads=6,
        drop_rate=drop,
        attn_drop_rate=drop,
        drop_path_rate=drop,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6)
    model = paddle.Model(net)
    model.load(r'pre param/63.pdparams')
    edge_blurs = EdgeBlurs()
    train_transform = T.Compose(
        [
            T.Resize(img_size),
            edge_blurs,
            T.RandomErasing(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                to_rgb=True,
            ),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                to_rgb=True,
            ),
        ]
    )

    train_dataset = Cifar10(mode='train', transform=train_transform, download=True, backend='cv2')
    # print(len(train_dataset))
    test_dataset = Cifar10(mode='test', transform=test_transform, download=True, backend='cv2')
    # print(len(test_dataset))

    epochs = 300
    warmup_epoch = 10

    # for img, label in itertools.islice(iter(train_dataset), 5):
    #     print(type(img), img.shape, label)

    lr = 1e-3
    # scheduler = CosineWarmup(lr=lr, step_each_epoch=100, epochs=epochs, warmup_epoch=warmup_epoch, start_lr = 0, end_lr = lr, verbose = True)
    # scheduler = LinearWarmup(learning_rate=lr, warmup_steps=warmup_epoch, start_lr=0, end_lr=lr)
    # scheduler = CosineAnnealingDecay(learning_rate=lr, T_max=warmup_epoch, eta_min=0)
    optimizer = Adam(learning_rate=lr, parameters=model.parameters())
    model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))

    print()
    print(f'the lr is {lr}')
    print(f'the epochs is {epochs}, and the warmup epoch is {warmup_epoch}')
    print()

    use_gpu = True
    paddle.set_device('gpu:0')
    model.fit(train_dataset,
              test_dataset,
              epochs=epochs,
              batch_size=300,
              save_dir="param",
              save_freq=7,
              log_freq=1,
              num_workers=0,
              shuffle=True,
              eval_freq=7,
              verbose=1)

    print("========end========")

    end_time = datetime.datetime.now()
    print(f'训练结束，当前时间为{end_time}')
