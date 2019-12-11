import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return x + self.model(x)


class IntModel(nn.Module):
    def __init__(self):
        super(IntModel, self).__init__()
        self.conv_A = self._build_conv()
        self.conv_B = self._build_conv()

        self.lstm_A = nn.LSTM(512, 256, 2, batch_first=True)
        self.lstm_B = nn.LSTM(512, 256, 2, batch_first=True)

        self.classifier = self._build_cls()

    def forward(self, x1, x2):
        bs = x1.shape[0]
        depth = x1.shape[1]
        chw = x1.shape[-3:]

        x1 = torch.reshape(x1, [-1] + list(chw))
        x2 = torch.reshape(x2, [-1] + list(chw))

        encoded_A = self.conv_A(x1)
        encoded_B = self.conv_B(x2)

        encoded_A = torch.reshape(encoded_A, (bs, depth, -1))
        encoded_B = torch.reshape(encoded_B, (bs, depth, -1))

        out_A, (h_A, c_A) = self.lstm_A(encoded_A)
        out_B, (h_B, c_B) = self.lstm_B(encoded_B)

        h_A = torch.reshape(torch.stack(torch.unbind(h_A, dim=1), dim=0), (bs, -1))
        h_B = torch.reshape(torch.stack(torch.unbind(h_B, dim=1), dim=0), (bs, -1))

        feat_AB = torch.cat((h_A, h_B), dim=1)

        prob = self.classifier(feat_AB)

        return prob

    def _build_cls(self, in_size=1024, num_cls=6):
        model = [nn.Linear(in_size, num_cls)]
        return nn.Sequential(*model)

    def _build_conv(self, in_ch=32, npf=64):
        n_blocks = 3
        model = [nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
                 nn.ReLU(True)]

        for cnt in range(n_blocks):
            previous = npf * 2 ** cnt
            model += [nn.Conv2d(previous, previous * 2, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(previous * 2),
                      nn.ReLU(True),
                      ResBlock(previous * 2, previous * 2)]

        model += [nn.AdaptiveMaxPool2d(1)]
        return nn.Sequential(*model)



if __name__ == '__main__':
    m = ResBlock(64,64)

    # x1 = torch.randn(4,5,32,64,64)
    # x2 = torch.randn(4,5,32,64,64)
    data = torch.randn(4,64,128,128)
    out = m(data)
    import ipdb
    ipdb.set_trace()
