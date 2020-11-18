# import os
# import cv2
# import numpy as np
# from collections import Counter
#
#
# def get_features(dir):
#     """
#     特征提取
#     :return:
#     """
#     row = 28//4
#     col = 28//4
#     imgs = [os.path.join(dir, i) for i in os.listdir(dir)]
#
#     features = np.zeros((len(imgs), 4*4+1), dtype=np.uint8)
#
#     for img in imgs:
#         # 读取图片
#         im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#         # 一张图片的特征
#         feature = np.zeros((1, 4 * 4 + 1), dtype=np.uint8)
#         # 提取图片特征
#         for row in range(4):
#             for col in range(4):
#                 # 获取一个特征
#                 box = im[col*7:(col+1)*7-1][row*7:(row+1)*7-1]
#                 cv2.imshow('im', box)
#                 cv2.waitKey()
#                 num = 0
#                 # 计算0出现的次数
#                 for i in box:
#                     for j in i:
#                         if j == 0:
#                             num += 1
#                 feature[0][(row+1)*(col+1)-1] = num
#         print(feature)
#
#     return
#
#
# if __name__ == '__main__':
#     path = os.getcwd() + '\\dataset\\0'
#     get_features(path)


# import cv2
#
# # 初始化数据
# width = 1200
# box_width = 200
# box_num = width//box_width
# # 单个样本的特征数量
# feature_count = box_num*box_num
#
# im = cv2.imread('xscn.jpg', cv2.IMREAD_GRAYSCALE)
# for col in range(box_num):
#     for row in range(box_num):
#         # box_im = im[col*box_width:(col+1)*box_width-1][row*box_width:(row+1)*box_width-1]
#         print(im.shape)
#         box_im = im[0:199, 1000:1199]
#         print(box_im.shape)
#         cv2.imshow('im', box_im)
#         cv2.waitKey()
#         cv2.destroyAllWindows()


import cv2
im = cv2.imread('test.bmp', cv2.IMREAD_GRAYSCALE)
r, rst = cv2.threshold(im, 210, 255, cv2.THRESH_BINARY)
# dst = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
# cv2.imshow('im', dst)
cv2.imshow('im', rst)
cv2.waitKey()
cv2.destroyAllWindows()