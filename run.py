import argparse
from importlib import import_module
from utils.from_maskJson_to_maskJpg import from_maskJson_to_maskJpg
from utils.split_wsi_to_patch import split_wsi_to_patch
from train import train


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

    """ 将语义分割标注的JSON格式文件转成jpg格式文件 """
    from_maskJson_to_maskJpg(config.jpg_save_dir, config.img_dir, config.json_dir, config.img_ext, config.json_ext, config.jpg_mask_ext, 
                             config.all_sample_reprocess)

    """ 将病理图像和标注文件从wsi级转成patch级 """
    split_wsi_to_patch(config.CROP_SIZE, config.STRIDE, config.IMG_SAVE_DIR, config.MASK_SAVE_DIR,config.img_dir, config.jpg_save_dir,
                        config.img_ext, config.jpg_mask_ext, config.patch_img_ext, config.patch_mask_ext,config.all_sample_reprocess)
    
    """ 模型训练 """
    train(config)
