import numpy as np
import cv2
import json
from PIL import Image
import glob
import os
from tqdm import tqdm


''' --- from_maskJson_to_maskJpg的目的 --- '''
""" 将语义分割标注的JSON格式文件转成jpg格式文件"""


# 解除解压缩炸弹保护限制
Image.MAX_IMAGE_PIXELS = None  # 或设置为一个足够大的整数

# 判断列表有几层
def nesting_depth(a_list):
    flag = False
    num = []
    for i in a_list:
        # 判断是否为队列
        if isinstance(i, list):
            flag = True
            # 判断队列中的元素是否是队列
            num.append(nesting_depth(i))
    if flag:
        return max(num) + 1
    else:
        return 0
    
# save_dir为获得MaskJpg的保存地址
# img_dir为WSI级别病理图像所在的文件夹
# json_dir为WSI级别病理图像的语义分割标注mask所在的文件夹,其格式为json
# img_ext为WSI级别病理图像的文件后缀
# json_ext为语义分割标注mask的json文件的后缀

def from_maskJson_to_maskJpg(jpg_save_dir, img_dir, json_dir, img_ext, json_ext, jpg_mask_ext,all_sample_reprocess):
    ''' --- 设置保存的路径 --- '''
    # 检查文件夹是否存在  
    if not os.path.exists(jpg_save_dir):  
        # 如果文件夹不存在，创建它  
        os.makedirs(jpg_save_dir)

    ''' --- 分别获取图片和标注列表 --- '''
    # 图片文件路径加载,并且进行排序
    img_list = sorted(glob.glob(os.path.join(img_dir, '*' + img_ext)))

    # 语义分割标注mask的json文件路径加载,并且进行排序
    json_list = sorted(glob.glob(os.path.join(json_dir, '*' + json_ext)))

    ''' --- 遍历Josn列表中每个json文件 --- '''
    # 遍历json文件路径队列
    for img_path, json_path in tqdm(zip(img_list, json_list)):

        # 打印文件路径
        print("img_path:" + img_path)
        print("json_path:" + json_path)

        # 获取文件名称
        img_name = img_path.split('/')[-1]
        # windows路径问题修正
        if "\\\\" in img_name:
            img_name = img_name.split('\\\\')[-1]
        if "\\" in img_name:
            img_name = img_name.split('\\')[-1]

        json_name = json_path.split('/')[-1]
        # windows路径问题修正
        if "\\\\" in json_name:
            json_name = json_name.split('\\\\')[-1]
        if "\\" in json_name:
            json_name = json_name.split('\\')[-1]

        #判断json与img是否匹配，如果不匹配，就下一个
        assert img_name[:-4] == json_name[:-5]

        # 设置语义分割标注的jpg格式文件的保存路径
        target_mask_path = os.path.join(jpg_save_dir, json_name[:-len(json_ext)] + jpg_mask_ext)

        # 是否重新处理所有文件
        if not all_sample_reprocess:
            if os.path.exists(target_mask_path):
                continue

        # 读取图片
        image = Image.open(img_path)

        #读取json文件
        with open(json_path, 'r') as f:
            json_dic = json.load(f)

        # 建立与图片文件等大小的0矩阵，其中[::-1]是调换元组中的两个元素的顺序
        mask = np.zeros([*image.size][::-1], dtype=np.float64)

        # 遍历json文件中'features'key对应的值
        for k, each in enumerate(json_dic['features'][:]):

            # 获取需要的队列
            coord_lists = each['geometry']['coordinates']

            # 没有孤岛 -- 第一个是外轮廓，其余是内轮廓
            if nesting_depth(coord_lists) == 2:  # 没有孤岛
                try:  # 进入这条分支说明是简单多边形
                    polygon = [np.array(coord_lists, np.int32)]
                except:  # 进入这条分支说明是复杂多边形
                    polygon = [np.array(item, np.int32) for item in coord_lists]
                # mask被填充相应的形状，并且填充值为K+1
                cv2.fillPoly(mask, polygon, k + 1)

            # 有孤岛 -- 每一个都有可能是Geometry
            elif nesting_depth(coord_lists) == 3:  # 有孤岛
                for list_id, coord_list in enumerate(coord_lists):
                    # 通过try except来判断是简单多边形还是复杂多边形
                    try:  # 进入这条分支说明是简单多边形
                        polygon = [np.array(coord_list, np.int32)]
                    except:  # 进入这条分支说明是复杂多边形
                        polygon = [np.array(item, np.int32) for item in coord_list]
                    cv2.fillPoly(mask, polygon, k + 1)

            # 语义分割标注的jpg格式文件
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(target_mask_path)




    

    



    

