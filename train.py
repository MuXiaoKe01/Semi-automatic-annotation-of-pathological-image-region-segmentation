import glob
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import numpy as np
import os
import datetime
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations import Resize
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms as abtransforms
import torch.backends.cudnn as cudnn
import cv2
import ssl  

# 程序可用显卡序号限制 
os.environ ['CUDA_VISIBLE_DEVICES'] = '0' 

# 当 cudnn.benchmark = True 时，PyTorch 会使用 cuDNN 的自动调优功能，
# 以便更好地利用 GPU 资源，提高深度神经网络的训练和推理速度。
cudnn.benchmark = True

# 解除解压缩炸弹保护限制
Image.MAX_IMAGE_PIXELS = None  # 或设置为一个足够大的整数

# 用于创建默认的 SSL（Secure Sockets Layer）上下文，以便在网络通信中加密数据传输。
ssl.create_default_context = ssl.create_default_context()

# 数据的分割
def split_list_into_k_parts(lst, k):

    n = len(lst)
    avg = n // k
    remainder = n % k

    result = []
    start = 0

    for i in range(k):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

# iou计算
def iou_score(output, target, num_classes):

    smooth = 1e-5

    if torch.is_tensor(output):

        # output = torch.sigmoid(output).data.cpu().numpy()
        # 转化成预测的最终结果，并且转化成narray类型
        output = torch.argmax(output, dim=1).cpu().numpy()

    if torch.is_tensor(target):

        # 转化成narray类型
        target = target.cpu().numpy()

        iou_scores = []
    
    #同时计算前景与背景
    for c in range(num_classes):

        # 像素点的交集数量
        intersection = np.sum((output == c) & (target == c))

        # 像素点的并集数量
        union = np.sum((output == c) | (target == c))
        if union == 0:
            # iou_scores.append(0.0)
            continue
        else:
            # iou公式
            iou_scores.append(intersection / union)

    # 求二者的均值
    mean_iou = np.mean(iou_scores)
    
    return mean_iou

# 构建dataSet
class SegDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None, patch_img_ext='.png', patch_mask_ext='.npy'):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_img_ext = patch_img_ext
        self.patch_mask_ext = patch_mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # 读取图片数据
        img = Image.open(os.path.join(self.img_dir, img_id + self.patch_img_ext)).convert('RGB')
        # 转化类型
        img = np.array(img)
        # 读取mask
        mask_path = os.path.join(self.mask_dir, img_id + self.patch_mask_ext)
        if self.patch_mask_ext == '.npy':
            mask = (np.load(mask_path) != 0).astype(np.uint8)
        else:
            mask = cv2.imread(mask_path, 0).astype(np.uint8)
        # 图像信息加强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        # 修正图片信息格式
        img = img.transpose(2, 0, 1)
        # mask = torch.tensor(mask[None, :, :]).float()

        # 转化类型
        mask = torch.tensor(mask).long()

        return img, mask, {'img_id': img_id}


def train_one_epoch(train_loader, model_ft, optimizer, criterion, device, num_classes):
    # 设置训练模式
    model_ft.train()
    # 设置初始的损失值和iou
    train_loss, train_iou = 0.0, 0.0

    # 开始训练
    with tqdm(train_loader, desc=' Training') as tbar:
        for imgs, masks, _ in tbar:

            # 将样本信息装入到设备中
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # 梯度归零
            optimizer.zero_grad()

            # 前向传播
            mask_outputs = model_ft(imgs)
            # 计算损失值
            loss = criterion(mask_outputs, masks)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()

            # 获取一个标量张量（scalar tensor）的 Python 值（float）
            step_loss = loss.item()

            # 累加损失值
            train_loss += step_loss * imgs.size(0)

            # 计算IOU
            iou = iou_score(mask_outputs, masks, num_classes)
            # 累加IOU
            train_iou += iou * imgs.size(0)
            # 更新打印出来的参数值
            tbar.set_postfix(iou=iou,
                             loss=step_loss)
            # 更新进度
            tbar.update()

        # 清除缓存
        torch.cuda.empty_cache()

    # 求最终损失的均值
    train_loss /= len(train_loader.dataset)
    # 求最终iou的均值
    train_iou /= len(train_loader.dataset)

    return train_loss, train_iou


def val_one_epoch(val_loader, model_ft, criterion, device, num_classes):
    # 设置验证模式
    model_ft.eval()
    # 设置初始的损失值和iou
    val_loss, val_iou = 0.0, 0.0
    with tqdm(val_loader, desc=' Val') as tbar:
        for imgs, masks, _ in tbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                mask_outputs = model_ft(imgs)
            loss = criterion(mask_outputs, masks)

            step_loss = loss.item()
            val_loss += step_loss * imgs.size(0)

            iou = iou_score(mask_outputs, masks, num_classes)
            val_iou += iou * imgs.size(0)

            tbar.set_postfix(iou=iou,
                             loss=step_loss)
            tbar.update()
        torch.cuda.empty_cache()

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)

    return val_loss, val_iou


def train(config):
 
    '''1. 创建数据加载器'''
    # 图像数据增强
    train_transform = A.Compose([
        # 转换成水平镜像图像，变换概率为p
        A.HorizontalFlip(p=config.horizontalFlip_p),
        # 转换成垂直镜像图像，变换概率为p
        A.VerticalFlip(p=config.verticalFlip_p),
        # 添加高斯噪声，变换概率为p
        A.GaussNoise(p=config.gaussNoise_p),
        # 用于于从一组可能的图像增强操作中随机选择一个来应用到图像上
        A.OneOf([
            A.MotionBlur(p=config.motionBlur_p),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=config.medianBlur_blur_limit, p=config.medianBlur_p),  # 中值滤波
            A.Blur(blur_limit=config.blur_blur_limit, p=config.blur_p),  # 使用随机大小的内核模糊输入图像。
        ], p=config.oneOf_p),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=config.randomBrightnessContrast_p),  # 随机明亮对比度
        # 调整图片的大小
        Resize(config.img_size, config.img_size),
        # 图像信息标准化
        abtransforms.Normalize(),
    ])

    val_transform = Compose([
        # 调整图片的大小
        Resize(config.img_size, config.img_size),
        # 图像信息标准化
        abtransforms.Normalize(),
    ])
    

    '''2. 创建数据加载器'''
    
    # 获得文件中的图片路径
    img_ids = glob.glob(os.path.join(config.IMG_SAVE_DIR, '*' + config.patch_img_ext))
    # 获取id
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # 将id分组
    KFolds = split_list_into_k_parts(img_ids, config.K)

    # 为适应windows文件夹的命名规则，将空格用短杆替代，把冒号改成短杆
    time_infor = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")

    # 获得模型的存储路径
    save_dir = os.path.join(config.exp_name, time_infor)
    
    # k折交叉实验
    for k, val_img_ids in enumerate(KFolds):

        # 训练样本
        train_img_ids = []
        for i in range(len(KFolds)):
            if i != k:
                train_img_ids += KFolds[i]

        # 制定dataset
        train_dataset, val_dataset = [SegDataset(
            img_ids=img_ids,
            img_dir=config.IMG_SAVE_DIR,
            mask_dir=config.MASK_SAVE_DIR,
            patch_img_ext=config.patch_img_ext,
            patch_mask_ext=config.patch_mask_ext,
            num_classes=config.n_classes,
            transform=data_trans) for img_ids, data_trans in [(train_img_ids, train_transform), (val_img_ids, val_transform)]]

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,# 加载数据的子进程个数
            drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4 * config.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False)
        
        '''3. 构建模型'''
        '''
        model_name\：要创建的模型的名称，例如 "unet"、"deeplabv3" 等。
        encoder_name\：用于提取特征的编码器\（backbone\）的名称，例如 "resnet34"、"resnet50" 等。
        in_channels\：输入图像的通道数。
        classes\：要分割的类别数。
        activation\：激活函数，通常是 "softmax" 或 "sigmoid"，用于输出分割结果。
        smp.create_model 函数返回一个语义分割模型，您可以使用这个模型进行图像分割任务。
        您可以使用不同的 model_name 和 encoder_name 组合来选择不同的预训练模型和架构。
        '''

        #加载模型结构
        model_ft = smp.create_model(arch=config.arch,
                                    encoder_name=config.encoder,
                                    classes=config.n_classes)  # MAnet, FPN, PSPNet

        # 设备选择
        device = torch.device(config.device)

        # 加载预训练模型
        if os.path.exists(config.ckpt_path):
            model_ft.load_state_dict(torch.load(config.ckpt_path))
        
        # 将模型加载到设备之中
        model_ft = model_ft.to(device)

        # 设置损失函数
        criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        '''
        这段代码使用了 PyTorch 中的 optim.SGD 优化器来训练一个深度学习模型。让我解释这段代码的各个部分：

        optim.SGD: 这是 PyTorch 中的随机梯度下降（Stochastic Gradient Descent，SGD）优化器，用于更新模型的权重以最小化损失函数。SGD 是深度学习中常用的优化算法之一。

        model_ft.parameters(): 这部分指定了要更新的模型参数。model_ft 是你的深度学习模型，而 parameters() 方法用于获取模型中所有需要训练的参数，例如权重和偏差。

        lr=0.001: 这是学习率（learning rate）的设置。学习率控制了权重更新的步长，较小的学习率会使权重更新更小步，较大的学习率则会使权重更新更大步。通常需要调整学习率来优化模型的训练。

        momentum=0.9: 这是动量（momentum）的设置。动量是 SGD 优化算法的一种变体，它有助于加速训练过程并减少震荡。动量的值通常设置在 0 到 1 之间，0.9 是一个常见的选择。
        '''
        optimizer = optim.SGD(model_ft.parameters(), lr=config.lr, momentum=config.momentum)

        """
            lr_scheduler.StepLR: 这部分指定了要使用的学习率调度器，即 StepLR 学习率调度器。学习率调度器是用于在训练期间动态调整学习率的工具。StepLR 调度器是一种基于步数的调度器，它在指定的步数（step_size 步数）时，将学习率乘以一个指定的因子（gamma）。

            optimizer: 这是你之前配置的优化器，通常是 optim.SGD 或其他优化器的实例。学习率调度器将调整此优化器的学习率。

            step_size=20: 这是调度器的步数，表示多少个训练步骤后会触发学习率的调整。在这个例子中，每经过 20 个训练步骤，学习率会按照下一个参数 gamma 的值进行调整。

            gamma=0.5: 这是学习率调度器的因子，表示每次触发学习率调整时，学习率将被乘以 0.5，即减小为原来的一半。这可以帮助在训练过程中逐渐减小学习率，从而有助于模型的收敛和稳定训练

        """
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

        # 模型拷贝
        best_model_wts = copy.deepcopy(model_ft.state_dict())

        #记录最后的情况下，准确率和iou
        best_index = {'acc': 0.0, 'iou': 0.0}

        '''4. 训练循环'''
        # 创建文件夹
        os.makedirs(os.path.join(save_dir, 'Fold_%d' % k), exist_ok=True)
        for epoch in range(config.num_epoch):
            # 训练，验证，输出，保存
            # 训练：数据，模型，优化器，损失函数，设备；损失、性能
            # 打印当前训练进度
            print('---- Epoch: %d / %d (k = %d) ----' % (epoch, config.num_epoch - 1, k))

            # 进行训练
            train_loss, train_iou = train_one_epoch(train_loader, model_ft, optimizer, criterion, device, config.n_classes)

            # 进行评估
            val_loss, val_iou = val_one_epoch(val_loader, model_ft, criterion, device, config.n_classes)
            print('train_loss = %.4f, train_iou = %.4f' % (train_loss, train_iou))
            print('val_loss = %.4f, val_iou = %.4f' % (val_loss, val_iou))
            # 更新学习率
            scheduler.step()
            
            # 如果训练出来的模型的评估效果好于之前，则进行记录
            if val_iou > best_index['iou']:

                best_index = {'iou': val_iou}
                best_model_wts = copy.deepcopy(model_ft.state_dict())
        # 打印相关信息               
        print('Fold_%d training finished, the best Iou is %.4f' % (k, best_index['iou']))

        #保存
        torch.save(best_model_wts, os.path.join(save_dir, 'Fold_%d' % k,'best_model_wts.pth'))