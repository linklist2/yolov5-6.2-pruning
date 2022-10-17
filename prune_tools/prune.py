# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s.xml                # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.yolo import Detect, Model
from val import run as valrun
from yolo_pruned import ModelPruned



from models.common import DetectMultiBackend, Bottleneck, Concat
from pruned_common import *
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           colorstr, increment_path, print_args)
from utils.torch_utils import select_device


def gather_bn_weights(module_list):
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights


def obtain_bn_mask(bn_module, thre, device):
    # thre = thre.cuda()
    thre = thre.to(device)
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


@torch.no_grad()
def run_prune(data,
              weights=None,  # model.pt path(s)
              cfg='models/yolov5l.yaml',
              percent=0,
              batch_size=32,  # batch size
              imgsz=640,  # inference size (pixels)
              task='val',  # train, val, test, speed or study
              device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
              workers=8,  # max dataloader workers (per RANK in DDP mode)
              single_cls=False,  # treat as single-class dataset
              save_txt=False,  # save results to *.txt
              project=ROOT / 'runs/val',  # save to project/name
              name='exp',  # save to project/name
              exist_ok=False,  # existing project/name ok, do not increment
              dnn=False,  # use OpenCV DNN for ONNX inference
              model=None,
              ):
    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model, 这里的fuse一定要关闭，否则BN层就被融合进Conv层，后面的BN层信息就无法统计了。
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fuse=False)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    data = check_dataset(data)  # check

    # Configure
    model = model.model
    model.eval()
    # =========================================== prune model ====================================#
    # print("model.module_list:",model.named_children())
    model_list = {}
    ignore_bn_list = []
    # 统计不需要剪枝的moudule
    for i, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            if layer.add:
                ignore_bn_list.append(i.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
    for i, layer in model.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            if i not in ignore_bn_list:
                model_list[i] = layer

    # 收集全部需要剪枝的BN层的权重
    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    # 为了使得每个BN层至少留一个通道，所以先查看BN层的最大值，然后从这些最大值里取一个最小值，这样就能保证每个BN层至少有一个没有被移除。
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())
    # print("highest_thre:",highest_thre)
    highest_thre = min(highest_thre)
    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(bn_weights)

    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}')

    assert percent_limit > percent, f'The pruning ratio should not exceed {percent_limit * 100:.3f}%!'

    # model_copy = deepcopy(model)
    # 根据指定的百分比获得剪枝阈值
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]
    print(f'Gamma value that less than {thre:.4f} are set to zero!')
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")

    # ============================== save pruned model config yaml =================================#
    remain_num = 0
    modelstate = model.state_dict()
    pruned_yaml = {}
    nc = model.model[-1].nc
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
    pruned_yaml["nc"] = model.model[-1].nc
    pruned_yaml["depth_multiple"] = model_yamls["depth_multiple"]
    pruned_yaml["width_multiple"] = model_yamls["width_multiple"]
    pruned_yaml["anchors"] = model_yamls["anchors"]
    anchors = model_yamls["anchors"]
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C3Pruned, [128]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C3Pruned, [256]],
        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        [-1, 9, C3Pruned, [512]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C3Pruned, [1024]],
        [-1, 1, SPPFPruned, [1024, 5]],  # 9
    ]
    pruned_yaml["head"] = [
        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C3Pruned, [512, False]],  # 13

        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C3Pruned, [256, False]],  # 17 (P3/8-small)

        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 14], 1, Concat, [1]],  # cat head P4
        [-1, 3, C3Pruned, [512, False]],  # 20 (P4/16-medium)

        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]],  # cat head P5
        [-1, 3, C3Pruned, [1024, False]],  # 23 (P5/32-large)

        [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
    ]

    # ============================================================================== #
    maskbndict = {}

    # 输出每一层的剪枝信息
    for bn_name, bn_layer in model.named_modules():
        if isinstance(bn_layer, nn.BatchNorm2d):
            if bn_name in ignore_bn_list:
                # mask = torch.ones(bn_layer.weight.data.size()).cuda()
                mask = torch.ones(bn_layer.weight.data.size()).to(device)
            else:
                mask = obtain_bn_mask(bn_layer, thre, device)

            maskbndict[bn_name] = mask
            # 当前层剩余的通道数
            layer_remain = int(mask.sum())
            assert layer_remain > 0, "Current remaining channel must greater than 0!!! " \
                                     "please set prune percent to lower thesh, or you can retrain a more sparse model..."
            # 统计总共还剩下多少通道
            remain_num += layer_remain

            # 将需要剪枝的BN层权重和偏置置0
            bn_layer.weight.data.mul_(mask)
            bn_layer.bias.data.mul_(mask)
            print(f"|\t{bn_name:<25}{'|':<10}{bn_layer.weight.data.size()[0]:<20}{'|':<10}{layer_remain:<20}|")

    print("=" * 94)

    # 为了避免上面的打印和下面的ModelPruned打印交叉，先在这里休眠1秒，等待上述内容完成
    time.sleep(1)

    # 根据mask信息重新构建yolo模型
    # pruned_model = ModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model = ModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).to(device)
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    # from_to_map存储每一层的输入来自于哪一层的信息,e.g.{'model.2.cv3.bn':['model.2.m.0.cv2.bn', 'model.2.cv2.bn']}
    from_to_map = pruned_model.from_to_map
    pruned_model_state = pruned_model.state_dict()
    assert pruned_model_state.keys() == modelstate.keys()
    # ===================================处理输入输出通道==================================================== #
    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(),
                                                                      pruned_model.named_modules()):
        assert layername == pruned_layername
        # model.24 是Detect层，其中有三个Conv
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            convname = layername[:-4] + "bn"
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()  # 处理输入通道

                    if len(w.shape) == 3:  # remain only 1 channel.
                        w = w.unsqueeze(1)
                    w = w[out_idx, :, :, :].clone()  # 处理输出通道

                    pruned_layer.weight.data = w.clone()  # 再次重新赋值
                    changed_state.append(layername + ".weight")
                if isinstance(former, list):
                    orignin = [modelstate[i + ".weight"].shape[0] for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(maskbndict[name].shape[0]) if maskbndict[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:, formerin, :, :].clone()
                    changed_state.append(layername + ".weight")
            else:
                # 处理model.0.Conv,因为该层没有来自上一层的输入，不会出现在from_to_map中
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w.clone()
                changed_state.append(layername + ".weight")

        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
            changed_state.append(layername + ".running_mean")
            changed_state.append(layername + ".running_var")
            changed_state.append(layername + ".num_batches_tracked")

        # 单独处理最后的Detect层
        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :]
            pruned_layer.bias.data = layer.bias.data
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")

    pruned_model.eval()
    pruned_model.names = model.names
    # =============================================================================================== #
    torch.save({"model": model}, os.path.join(save_dir, "orign_model.pt"))
    torch.save({"model": pruned_model}, os.path.join(save_dir, "pruned_model.pt"))
    LOGGER.info(f'Pruned model weights saved at {os.path.join(save_dir, "pruned_model.pt")}')

    # 开始验证剪枝后的模型的指标
    model = pruned_model
    model.to(device).eval()
    # model.cuda().eval()

    # 创建dataloader
    pad = 0.0 if task in ('speed', 'benchmark') else 0.5
    rect = False if task == 'benchmark' else pt  # square inference for benchmarks
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   stride,
                                   single_cls,
                                   pad=pad,
                                   rect=rect,
                                   workers=workers,
                                   prefix=colorstr(f'{task}: '))[0]

    data_dict = check_dataset(data)  # check if None
    LOGGER.info('After Pruning.....')
    valrun(
        data=data_dict,
        model=model,
        dataloader=dataloader,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        workers=opt.workers,
        half=opt.half,
        plots=False
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/mask.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5n_mask.yaml', help='model.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp6/weights/last.pt',
                        help='model path(s)')
    parser.add_argument('--percent', type=float, default=0.4, help='prune percentage')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')

    LOGGER.info('Before Pruning.....')

    # valrun(
    #     data=opt.data,
    #     weights=opt.weights,
    #     batch_size=opt.batch_size,
    #     imgsz=opt.imgsz,
    #     workers=opt.workers,
    #     project=opt.project,
    #     half=opt.half
    # )

    LOGGER.info('Pruning.....')
    run_prune(
        data=opt.data,
        weights=opt.weights,
        cfg=opt.cfg,
        percent=opt.percent,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        workers=opt.workers,
        project=opt.project
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
