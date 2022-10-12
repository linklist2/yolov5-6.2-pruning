import math
from copy import deepcopy
from pathlib import Path

from models.common import Concat, Contract, Expand
from models.yolo import BaseModel, Detect, Segment
from pruned_common import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER
from utils.torch_utils import initialize_weights, scale_img


def parse_pruned_model(maskbndict, d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    fromlayer = []  # last module bn layer name
    from_to_map = {}
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m in [Conv]:
            named_m_bn = named_m_base + ".bn"

            bnc = int(maskbndict[named_m_bn].sum())
            c1, c2 = ch[f], bnc
            args = [c1, c2, *args[1:]]
            layertmp = named_m_bn
            if i > 0:
                from_to_map[layertmp] = fromlayer[f]
            fromlayer.append(named_m_bn)

        elif m in [C3Pruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            named_m_cv3_bn = named_m_base + ".cv3.bn"
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = fromlayer[f]
            fromlayer.append(named_m_cv3_bn)

            cv1in = ch[f]
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            cv3out = int(maskbndict[named_m_cv3_bn].sum())
            args = [cv1in, cv1out, cv2out, cv3out, n, args[-1]]
            bottle_args = []
            chin = [cv1out]

            c3fromlayer = [named_m_cv1_bn]
            for p in range(n):
                named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(p)
                named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(p)
                bottle_cv1in = chin[-1]
                bottle_cv1out = int(maskbndict[named_m_bottle_cv1_bn].sum())
                bottle_cv2out = int(maskbndict[named_m_bottle_cv2_bn].sum())
                chin.append(bottle_cv2out)
                bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
                from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[p]
                from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                c3fromlayer.append(named_m_bottle_cv2_bn)
            args.insert(4, bottle_args)
            c2 = cv3out
            n = 1
            from_to_map[named_m_cv3_bn] = [c3fromlayer[-1], named_m_cv2_bn]
        elif m in [SPPFPruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            cv1in = ch[f]
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn] * 4
            fromlayer.append(named_m_cv2_bn)
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)
        elif m is Detect:
            from_to_map[named_m_base + ".m.0"] = fromlayer[f[0]]
            from_to_map[named_m_base + ".m.1"] = fromlayer[f[1]]
            from_to_map[named_m_base + ".m.2"] = fromlayer[f[2]]
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
            fromtmp = fromlayer[-1]
            fromlayer.append(fromtmp)

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), from_to_map


class ModelPruned(BaseModel):
    # YOLOv5 detection model
    def __init__(self, maskbndict, cfg='yolov5s.yaml', ch=3, nc=None,
                 anchors=None):  # model, input channels, number of classes
        super().__init__()
        self.maskbndict = maskbndict
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 与原本的DetectionModel只有在解析模型时不同，其余均相同
        self.model, self.save, self.from_to_map = parse_pruned_model(self.maskbndict, deepcopy(self.yaml),
                                                                     ch=[ch])  # model, savelist

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
