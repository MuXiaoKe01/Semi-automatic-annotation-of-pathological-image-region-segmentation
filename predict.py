import argparse
import os
from glob import glob
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import torch
import torch.backends.cudnn as cudnn
from albumentations.augmentations import transforms
from tqdm import tqdm
import numpy as np

import segmentation_models_pytorch as smp
from importlib import import_module
import torchvision.transforms as transforms_
import random
import json
import ssl 

# 程序可用显卡序号限制 
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

# 当 cudnn.benchmark = True 时，PyTorch 会使用 cuDNN 的自动调优功能，
# 以便更好地利用 GPU 资源，提高深度神经网络的训练和推理速度。
cudnn.benchmark = True

# 解除解压缩炸弹保护限制
Image.MAX_IMAGE_PIXELS = None  # 或设置为一个足够大的整数

# 用于创建默认的 SSL（Secure Sockets Layer）上下文，以便在网络通信中加密数据传输。
ssl.create_default_context = ssl.create_default_context()

# 去掉小轮廓
def remove_small_objs(array_in, min_size):
    """
    Removes small foreground regions from binary array, leaving only the contiguous regions which are above
    the size threshold. Pixels in regions below the size threshold are zeroed out.

    Args:
        array_in (np.ndarray): Input array. Must be binary array with dtype=np.uint8.
        min_size (int): Minimum size of each region.

    Returns:
        np.ndarray: Array of labels for regions above the threshold. Each separate contiguous region is labelled with
            a different integer from 1 to n, where n is the number of total distinct contiguous regions
    """
    # 判断类型是否为np.uint8
    assert (
            array_in.dtype == np.uint8
    ), f"Input dtype is {array_in.dtype}. Must be np.uint8"
    # remove elements below size threshold
    # each contiguous nucleus region gets a unique id
    """
    cv2.connectedComponents(array_in) 是 OpenCV（Open Source Computer Vision Library）
    中的一个函数，用于执行连通组件标记（Connected Components Labeling）操作。这个函数的作用是将
    输入的二进制数组（通常是二值化的图像）中的连通区域分配不同的标签，以便识别和分析各个物体或区域。

    以下是一些主要的参数和功能：

    array_in: 这是输入的二进制数组，通常是一个包含像素值为0和1的图像或区域。该数组中的0通常表示背景，
    1表示前景对象，或者其他类似的二进制值。connectedComponents 函数将在这个数组上执行标签操作。
    返回值：

    connectedComponents 函数返回一个元组，其中包含两个元素：
    第一个元素是标签化后的数组，其中每个连通区域都被赋予一个唯一的整数标签。这个数组的形状与输入数组相同，
    但每个像素点都有一个标签，用于表示它所属的连通区域。
    第二个元素是一个整数，表示连接组件的总数（包括背景）
    """
    n_labels, labels = cv2.connectedComponents(array_in)
    # each integer is a different nucleus, so bincount gives nucleus sizes

    """
    labels.flatten()将数组转一维
    np.bincount()用于对标签的个数进行计数
    """
    sizes = np.bincount(labels.flatten())
    
    # n_labels表示有几种标签，sizes表示该类标签的总数
    for nucleus_ix, size_ix in zip(range(n_labels), sizes):
        # 如果该类标签的个数小于阈值
        if size_ix < min_size:
            # below size threshold - set all to zero
            # 将该类标签设为0
            labels[labels == nucleus_ix] = 0

    # 重新对连通区域分配不同的标签
    _, labels = cv2.connectedComponents(labels.astype(np.uint8))  

    return labels

# 将预测结果转换成JSON格式
def convert_results_to_json(coord_list):
    def convert_coord_to_anno(coord):

        # 遍历每个轮廓
        for k in range(len(coord)):
            # 是否一个轮廓中的起点和终点的坐标一样，如果不同，在列表尾部加入起点坐标
            if coord[k][-1] != coord[k][0]:
                coord[k].append(coord[k][0])

        # 字典嵌套
        geometry = {'coordinates': coord,
                    'type': 'Polygon'}
        anno = {'geometry': geometry,
                'properties': {'isLocked': False,
                               'object_type': 'annotation'},
                'type': 'Feature'}
        return anno
    
    # 构建字典
    json_dic = {'type': 'FeatureCollection',
                'features': []}
    
    # 将每个区域的信息记录到字典中
    for coord in coord_list:
        # if len(coord) > 5:
        # 外轮廓像素点个数大于五
        if len(coord[0]) > 5:
            anno = convert_coord_to_anno(coord)
            json_dic['features'].append(anno)
    return json_dic


def predict(config):

    """
    cudnn.benchmark 是 PyTorch 中与 GPU 计算性能优化相关的一个设置。cudnn 是 NVIDIA 提供的用于深度学习的 GPU 加速库，
    用于加速深度神经网络的训练和推理。当你将 cudnn.benchmark 设置为 True 时，PyTorch 会尝试优化网络性能以减少每个批次的计算时间。
    """
    cudnn.benchmark = True

    # 模型结构加载
    model = smp.create_model(arch=config.arch,
                             encoder_name=config.encoder,
                             classes=config.n_classes)  # MAnet, FPN, PSPNet

    # 模型权重加载
    state_dict = torch.load(config.predict_ckpt_path, map_location=config.device)

    #将模型权重加载到模型结构中
    try:
        model.load_state_dict(state_dict)
    except:
        _state_dict = {}
        for key in state_dict:
            if key.startswith("model"):
                _state_dict[key[len("model."):]] = state_dict[key]
            else:
                _state_dict[key] = state_dict[key]
        model.load_state_dict(_state_dict, strict=False)
        del state_dict

    # 设备选择
    device = torch.device(config.device)
    
    # 将模型加载到设备之中
    model = model.to(device)

    # 进入评估模式
    model.eval()

    ''' -------- infer selected images -------- '''
    
    # 读取文件夹中的图片文件路径
    img_paths = glob(os.path.join(config.predict_img_dir, '*.jpg'))
    # 对图片文件路径顺序进行打乱
    random.shuffle(img_paths)

    # 存储json的位置
    json_save_dir = config.predict_json_save_dir

    # 存储jpg的位置
    jpg_save_dir = config.predict_jpg_save_dir

    # 创建相应的文件夹
    os.makedirs(json_save_dir, exist_ok=True)
    
    # 遍历图片文件路径
    for img_path in tqdm(img_paths[:]):
        img_name = img_path.split('/')[-1]

        # windows路径问题修正
        if "\\\\" in img_name:
            img_name = img_name.split('\\\\')[-1]
        if "\\" in img_name:
            img_name = img_name.split('\\')[-1]


        #设置图片和mask存储地址
        save_path = os.path.join(json_save_dir, '%s.json' % img_name[:-len('.jpg')])
        mask_save_path = os.path.join(jpg_save_dir, '%s.jpg' % img_name[:-len('.jpg')])

        #如果已经预测过，将不再预测
        if os.path.exists(mask_save_path):
            continue

        # 读取图片
        img = Image.open(img_path).convert('RGB')

        # 获取图片的宽和高
        W, H = img.size

        ''' --------- 切成小图输入 --------- '''

        #网格切割图像相关参数
        CROP_SIZE = config.predict_CROP_SIZE
        STRIDE_SIZE = config.predict_STRIDE_SIZE

        # 预留交叉区域
        OFFSET = (CROP_SIZE - STRIDE_SIZE) // 2

        # 分割块数
        kw = int(np.ceil((W - STRIDE_SIZE) / STRIDE_SIZE) + 1)
        kh = int(np.ceil((H - STRIDE_SIZE) / STRIDE_SIZE) + 1)

        # 这里用于填充的空间可能会比实际的图片的大小要大
        output_tensor = np.zeros((kh * STRIDE_SIZE, kw * STRIDE_SIZE))

        # 进行分割
        for w in range(kw):
            for h in range(kh):
                #从图片中，对区域进行
                cropped_img = img.crop((w * STRIDE_SIZE - OFFSET, h * STRIDE_SIZE - OFFSET,
                                        (w + 1) * STRIDE_SIZE + OFFSET,
                                        (h + 1) * STRIDE_SIZE + OFFSET))  # (left, upper, right, lower)

                # 图像数据类型转化以及标准化
                img_np = transforms.Normalize()(image=np.array(cropped_img))['image']

                # 图像数据类型转化并且在前面加一个维度
                img_tensor = transforms_.ToTensor()(img_np).to(device).unsqueeze(0)

                # 获取预测结果
                with torch.no_grad():
                    model_out = model(img_tensor)[0, :, OFFSET: OFFSET+STRIDE_SIZE, OFFSET: OFFSET+STRIDE_SIZE]
                
                # 得到预测结果
                model_out = torch.argmax(model_out, dim=0).cpu().numpy()

                # 去除比较小的区域
                model_out = (remove_small_objs(model_out.astype(np.uint8), min_size=500) > 0) * model_out
                
                # 填充回大图之中
                output_tensor[h * STRIDE_SIZE:(h + 1) * STRIDE_SIZE, \
                    w * STRIDE_SIZE:(w + 1) * STRIDE_SIZE] = model_out
        
        # 获取原图片大小的预测
        output = output_tensor[:H, :W]

        # 保存mask图像
        Image.fromarray((output > 0).astype(np.uint8) * 255).save(mask_save_path)
        

        # ----------- export to json file -----------

        # 重新对连通区域分配不同的标签
        _, np_preds = cv2.connectedComponents(output.astype(np.uint8))
        coord_list = []

        # 遍历除0之外的其他标签
        for label in np.unique(np_preds)[1:]:
            nuclei_mask = np_preds == label
            """
                cv2.RETR_TREE: 这是轮廓检索模式的一个参数。cv2.RETR_TREE 表示检索所有轮廓并创建一个完整的层次结构树。
                层次结构树描述了轮廓之间的嵌套关系，以及它们的父子关系。这个信息对于分析复杂的轮廓结构非常有用。

                cv2.CHAIN_APPROX_NONE: 这是轮廓近似方法的一个参数。cv2.CHAIN_APPROX_NONE 表示不对轮廓进行近似，
                保留所有轮廓点。如果你需要保留轮廓的每个像素点，这是一个合适的选项。

                cv2.findContours函数返回一个包含所有轮廓信息的列表。每个轮廓由一系列点坐标组成，
                你可以使用这些坐标来绘制或进一步分析轮廓。
            
            
            """
            contours, hierarchy = cv2.findContours(nuclei_mask.astype(np.uint8),
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)
            """
                使用 squeeze() 可以将轮廓数据从形状为 (1, n, 2) 的数组压缩为形状为 (n, 2) 的数组，
            """
            coord = [contour.squeeze().tolist() for contour in contours]

            # 加入到轮廓队列中
            coord_list.append(coord)

        # 将结果转化成字典
        json_dic = convert_results_to_json(coord_list)

        # 将字典转化成json格式
        dumped = json.dumps(json_dic)

        # 保存json
        with open(save_path, 'w') as f2:
            f2.write(dumped)
        # # -------------------------------------------
    # 释放资源
    torch.cuda.empty_cache()


# 输入相关参数
parser = argparse.ArgumentParser(description = "segmentation_model_V1")
parser.add_argument('--config', default='config_v1', type=str, help="配置文件路径")
args = parser.parse_args()

if __name__ == "__main__":


    """ 配置文件加载 """
    # 读取配置文件
    config_file = import_module("config." + args.config)

    # 创建配置对象
    config = config_file.Config()

    # 预测
    predict(config)
