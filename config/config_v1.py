import torch


class Config(object):

    """配置参数"""
    def __init__(self):

        ''' --- 数据预处理配置参数 --- '''

        # all_sample_reprocess是重新处理所有文件，如果是，则重新处理所有文件，否则只处理里新添加的样本
        self.all_sample_reprocess = False
        
        # save_dir为获得WSI级别MaskJpg的保存地址
        self.jpg_save_dir = r"data/breast_cancer_pathological_image_v1/annotated_samples/WSI/mask_jpg"

        # img_dir为WSI级别病理图像所在的文件夹
        self.img_dir = r"data/breast_cancer_pathological_image_v1/annotated_samples/WSI/samples_photo"

        # json_dir为WSI级别病理图像的语义分割标注mask所在的文件夹,其格式为json
        self.json_dir = r"data/breast_cancer_pathological_image_v1/annotated_samples/WSI/mask_json"

        # IMG_SAVE_DIR为转成patch级的病理图像的存储位置
        self.IMG_SAVE_DIR = r"data/breast_cancer_pathological_image_v1/annotated_samples/patch/samples_photo"

        # MASK_SAVE_DIR为转成patch级的标注文件npy格式文件的存储位置
        self.MASK_SAVE_DIR = r"data/breast_cancer_pathological_image_v1/annotated_samples/patch/mask_npy"

        # img_ext为WSI级别病理图像的文件后缀
        self.img_ext = r".jpg"

        # json_ext为WSI级别语义分割标注mask的json文件的后缀
        self.json_ext = r".json"
        
        # jpg_mask_ext为WSI级别语义分割标注mask的jpg文件的后缀
        self.jpg_mask_ext = r".jpg"

        # patch_img_ext为patch级别病理图像的文件后缀
        self.patch_img_ext = r".png"

        # patch__mask_ext为patch级别语义分割标注mask文件的后缀
        self.patch_mask_ext = r".npy"

        # 网格状切割的相关参数
        self.CROP_SIZE = 1024

        # 网格状切割的相关参数
        self.STRIDE = 512


        ''' --- 图像数据增强配置参数 --- '''

        # 转换成水平镜像图像的变换概率
        self.horizontalFlip_p = 0.5

        # 转换成垂直镜像图像的变换概率
        self.verticalFlip_p = 0.5

        # 添加高斯噪声的变换概率
        self.gaussNoise_p = 0.5

        # 用于于从一组可能的图像增强操作中随机选择一个来应用到图像上的变换概率
        self.oneOf_p = 0.2

        # 使用随机大小的内核将运动模糊应用于输入图像的变换概率
        self.motionBlur_p = 0.2

        # 中值滤波核的大小
        self.medianBlur_blur_limit = 3

        # 中值滤波的变换概率
        self.medianBlur_p = 0.1

        # 随机大小的内核模糊输入图像核的大小
        self.blur_blur_limit = 3

        # 随机大小的内核模糊输入图像的变换概率
        self.blur_p = 0.1

        # 随机明亮对比度的变换概率
        self.randomBrightnessContrast_p = 0.2


        ''' --- 训练模型配置参数 --- '''

        # 模型训练设备
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 类别数量
        self.n_classes = 2

        # 训练次数
        self.num_epoch = 30

        # 图片大小
        self.img_size = 512

        # batch大小
        self.batch_size = 16

        # 模型权重最终的存储位置
        self.exp_name = r"model/state_dict/breast_cancer_pathological_image_v1"

        # 模型名称
        self.arch = 'Unet'

        # 加密模型
        self.encoder = 'resnet34'

        # 将样本分为k组,用于k折交叉实验
        self.K = 5

        # 预训练模型
        self.ckpt_path = r""

        # 学习率
        self.lr = 0.001

        # 动量
        self.momentum = 0.9

        # 调度器的步数，表示多少个训练步骤后会触发学习率的调整。
        self.step_size = 20

        # 这是学习率调度器的因子，表示每次触发学习率调整时，学习率将被乘以gamma
        self.gamma = 0.5

        ''' --- 预测配置参数 --- '''

        # 参与预测的未标注病理图像所在的文件夹
        self.predict_img_dir = r"data/breast_cancer_pathological_image_v1/unannotated_samples/photo"

        # 预测模型权重
        self.predict_ckpt_path = r"model/state_dict/breast_cancer_pathological_image_v1/2023-11-11-10-26-36.402200/Fold_4/best_model_wts.pth"

        # 存储预测的语义分割结果json文件的所在文件夹位置
        self.predict_json_save_dir = r'data/breast_cancer_pathological_image_v1/unannotated_samples/predict_mask_json'

        # 存储预测的语义分割结果jpg文件的所在文件夹位置
        self.predict_jpg_save_dir = r"data/breast_cancer_pathological_image_v1/unannotated_samples/predict_mask_jpg"

        # 预测网格状切割的相关参数
        self.predict_CROP_SIZE = 2048

        # 预测网格状切割的相关参数
        self.predict_STRIDE_SIZE = 2000








       


        