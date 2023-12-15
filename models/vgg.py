import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, act_cls=None, pool='max'):
        r'''
        
        Args:
            vgg_name (str): one of 'VGG11', 'VGG13', 'VGG16', 'VGG19'
            act_cls (nn.Module, optional): activation class, e.g. nn.ReLU, nn.LeakyReLU, nn.PReLU, etc.
            pool (str, optional): one of 'max', 'strided'. Default: 'max'
        
        Notes on activation:
            The activations are constructed without any arguments, i.e. act_cls().
            If arguments are needed, use `functools.partial` to create a new class.
        
        Notes on pooling:
            If the pooling is 'max' the pooling layers are `nn.MaxPool2d` with kernel size 2 and stride 2.
            If the pooling is 'strided', the pooling layers are identity, but the conv layers have stride 2.
        
        '''
        super().__init__()

        self.act_cls = act_cls
        if self.act_cls is None:
            self.act_cls = nn.ReLU
        
        self.conv_extra_kwargs = {}
        if callable(pool):
            self.pool_cls = pool
        else:
            assert pool in ('max', 'strided')
            if pool == 'max':
                self.pool_cls = nn.MaxPool2d
            else:
                self.pool_cls = nn.Identity
                self.conv_extra_kwargs['stride'] = 2

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.pool_cls(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, **self.conv_extra_kwargs),
                           nn.BatchNorm2d(x),
                           self.act_cls()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
