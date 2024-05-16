import glob
import os
import numpy as np
from PIL import Image
from tqdm import trange


''' --- split_wsi_to_patch的目的 --- '''
""" 将病理图像和标注文件从wsi级转成patch级 """
"""         转化方式为网格切割            """


# 解除解压缩炸弹保护限制
Image.MAX_IMAGE_PIXELS = None  # 或设置为一个足够大的整数

# CROP_SIZE,STRIDE为网格状切割的相关参数
# IMG_SAVE_DIR为转成patch级的病理图像的存储位置
# MASK_SAVE_DIR为转成patch级的标注文件npy格式文件的存储位置
# IMG_DIR为WSI级的病理图像的存储位置
# MASK_DIR为WSI级的标注文件jpg格式文件的存储位置
# img_ext为WSI级别病理图像的文件后缀
# jpg_mask_ext为WSI级别语义分割标注mask的jpg文件的后缀
# patch_img_ext为patch级别病理图像的文件后缀
# patch_mask_ext为patch级别语义分割标注mask文件的后缀
def split_wsi_to_patch(CROP_SIZE, STRIDE, IMG_SAVE_DIR, MASK_SAVE_DIR, img_dir, jpg_save_dir, img_ext, jpg_mask_ext, patch_img_ext, patch_mask_ext,all_sample_reprocess):

    #创建相应的文件夹
    if not os.path.exists(IMG_SAVE_DIR):  
        # 如果文件夹不存在，创建它  
        os.makedirs(IMG_SAVE_DIR)

    #创建相应的文件夹
    if not os.path.exists(MASK_SAVE_DIR):  
        # 如果文件夹不存在，创建它  
        os.makedirs(MASK_SAVE_DIR)

    
    # 扫描文件夹中的文件
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*' + img_ext)))
    mask_path_list = sorted(glob.glob(os.path.join(jpg_save_dir, '*' + jpg_mask_ext)))

    # 对文件进行处理
    for img_path, mask_path in zip(img_path_list, mask_path_list):

        # 获取文件名称
        img_name = img_path.split('/')[-1][:-len(img_ext)]
        mask_name = mask_path.split('/')[-1][:-len(jpg_mask_ext)]

        # windows路径问题修正
        if "\\\\" in img_name:
            img_name = img_name.split('\\\\')[-1]
        if "\\" in img_name:
            img_name = img_name.split('\\')[-1]

        # windows路径问题修正
        if "\\\\" in mask_name:
            mask_name = mask_name.split('\\\\')[-1]
        if "\\" in mask_name:
            mask_name = mask_name.split('\\')[-1]

        # 判断是否是图片和mask是否是一一对应
        assert img_name == mask_name

        # 打开文件
        img = Image.open(img_path)
        mask = np.array(Image.open(mask_path))

        # 判断图片与mask大小是否一致
        assert list(img.size)[::-1] == list(mask.shape)

        #获得图片的宽和高
        W, H = img.size

        # 横切块数
        kw = int((W - CROP_SIZE) / STRIDE) + 1

        # 纵切块数
        kh = int((H - CROP_SIZE) / STRIDE) + 1

        # trange()可以进度条显示
        # 遍历每一个patch
        for w in trange(kw):
            for h in range(kh):

                # 设置文件存储位置
                img_save_path = os.path.join(IMG_SAVE_DIR, ('%s_x%d_y%d' + patch_img_ext) % (img_name, w, h))
                mask_save_path = os.path.join(MASK_SAVE_DIR, ('%s_x%d_y%d' + patch_mask_ext) % (mask_name, w, h))

                # 是否重新处理所有文件
                if not all_sample_reprocess:
                    if os.path.exists(img_save_path):
                        break

                # 图片的区域截取
                cropped_img = img.crop((w * CROP_SIZE, h * CROP_SIZE,
                                    (w + 1) * CROP_SIZE, (h + 1) * CROP_SIZE))  # (left, upper, right, lower)
                # mask的区域截取
                cropped_mask = mask[h * CROP_SIZE:  (h + 1) * CROP_SIZE, w * CROP_SIZE: (w + 1) * CROP_SIZE]

                # 截取出错
                if not cropped_mask.shape == (CROP_SIZE, CROP_SIZE):
                    continue

                # 截取区域没有进行标注             
                if np.mean(cropped_mask) == 0:  # add at 20.03.12
                    continue
                #     if np.random.random() < 0.9:
                #         continue

                # 将截取得到的结果进行保存
                
                cropped_img.save(img_save_path, quality=95)
                np.save(mask_save_path, cropped_mask)




    

    
