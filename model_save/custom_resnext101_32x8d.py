from torch import nn


class Custom_ResNeXt101_32x8d(nn.Module):
    def __init__(self, model, out_c):
        """
        model是已经加载预训练参数的模型
        """
        super(Custom_ResNeXt101_32x8d, self).__init__()
        # 去掉model最后三层，同时固定剩余层参数
        self.resnext_layer = nn.Sequential(*list(model.children())[:-3])
        for params in self.resnext_layer.parameters():
            params.requires_grad = False

        # 附加的自定义层
        self.cus_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1024, out_c, bias=False),
        )

    def forward(self, x):
        x = self.resnext_layer(x)
        x = self.cus_layer(x)
        return x


if __name__ == "__main__":
    pass
