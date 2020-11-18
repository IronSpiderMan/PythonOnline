import os
import cv2
import pickle
import numpy as np
import pandas as pd


def get_features():
    # 初始化数据
    width = 28
    box_width = 7
    box_num = width // box_width
    # 单个样本的特征数量
    feature_count = box_num * box_num

    # 获取图片路径列表
    base_dirs = [os.path.join(os.getcwd(), 'dataset', '%s' % i) for i in range(10)]
    image_lists = [[os.path.join(base_dir, file) for file in os.listdir(base_dir)] for base_dir in base_dirs]

    # 总共的图片数，样本数量
    image_count = 0
    for image_list in image_lists:
        for image in image_list:
            image_count += 1

    # 特征数组
    features = np.zeros((image_count, feature_count + 1), dtype=np.uint8)

    features_num = 0
    # 特征提取
    for image_list in image_lists:
        for image in image_list:
            # 读取图片，并进行二值化处理
            im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            r, im = cv2.threshold(im, 210, 255, cv2.THRESH_BINARY)
            # 单个图片的特征值
            feature = np.zeros((1, feature_count + 1), dtype=np.uint8)
            current_feature_num = 0
            # 提取图片特征
            for col in range(box_num):
                for rol in range(box_num):
                    box_im = im[col * box_width:(col + 1) * box_width, rol * box_width:(rol + 1) * box_width]
                    # 计算0出现的次数
                    times = 0
                    for i in box_im:
                        for j in i:
                            if j == 0:
                                times += 1
                    feature[0][current_feature_num] = times
                    current_feature_num += 1
            target = image.split('dataset\\')[1].split('\\')[0]
            feature[0][-1] = target
            features[features_num] = feature
            features_num += 1
    return features


if __name__ == '__main__':
    if os.path.exists('./features.pkl'):
        # 加载特征
        features = pickle.load(open('./features.pkl', 'rb'))

        # 准备用于识别的图片

    else:
        print("文件不存在")
        features = get_features()
        print(features, features.shape)
        df = pd.DataFrame(features)
        df.to_csv('./features.csv')
        pickle.dump(features, open('./features.pkl', 'wb'))
