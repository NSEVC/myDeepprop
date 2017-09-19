#coding:utf-8
import cv2
import argparse
import json, time
import scipy.sparse.linalg
import sklearn.feature_extraction
import myDNN
import numpy as np
from skimage import segmentation
from keras.utils import np_utils

class DeepProp(object):
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.patch_radius = 4
        self.t_data_ratio = 0.1
        self.sp_ratio = 0.01

        self.dnn = myDNN.myDNN(patch_radius=self.patch_radius)


    def recoloring(self, strk_path, out_path="../data/out.jpg"):
        # 1. get data
        # (1) get the original img
        print("load input...")
        img = self.img
        assert img.any(),"Can not read the image!"

        # (2) get the strk img
        strk = cv2.imread(strk_path, -1)  # 为什么加了-1之后读入的是四个通道？？
        assert strk.any(), "Can not read the strk image!"

        # (3) get the train data from original img
        x_train = list()
        x_train_coord = list()
        y_train = list()

        color2label = dict()
        patch_radius = 4
        h, w = img.shape[:2]

        reflected_img = cv2.copyMakeBorder(img, patch_radius, patch_radius, patch_radius,patch_radius, cv2.BORDER_REFLECT_101)
        # 对扩充后的图像/255. , 再提取9*9的patch
        patch_features = sklearn.feature_extraction.image.extract_patches_2d(reflected_img / 255., (patch_radius * 2 + 1, patch_radius * 2 + 1))

        for y, row in enumerate(strk[:, :, 3]):
            for x, val in enumerate(row):
                if val == 0:  # ???定位到有笔触的像素点x,y--第四通道透明度为0或者？
                    continue
                color_str = json.dumps(strk[y, x][:3].tolist())  # 提取了笔触图片的行，将其转为列表并进行json格式化编码
                color2label[color_str] = color2label.get(color_str, len(color2label.keys()))  # 颜色RGBjson格式作为字典的键与值（标签）0,1,...对应
                x_train.append(np.asarray(patch_features[y * w + x]))  # 定位到像素点对应的patch
                x_train_coord.append([y / float(h - 1), x / float(w - 1)])  # 以0,0为原点的相对坐标
                y_train.append(color2label[color_str])  # 通过键索引，将标签0,1,...放入Y_label的列表中
                # 将颜色标签的键值对互换
                label2color = {v: np.array(json.loads(k), np.float) for k, v in color2label.items()}
        # print(x_train)
        # print(x_train_coord)

        # (4) get the test data fome the original img. Here we use the SILC to get proper test data
        # SILC分割结果;输出一个314*400的矩阵（list）（与图像大小相等），里面的每一个值表示该像素点属于哪一个超像素块
        segments = segmentation.slic(img, n_segments=int(h * w * 0.01), compactness=50.,
                                             enforce_connectivity=True)

        spid2center = dict()
        spid2pixnum = dict()
        for y in range(h):
            for x in range(w):
                spid2center[segments[y, x]] = spid2center.get(segments[y, x],np.zeros(2, np.float)) \
                                              + np.array([y, x],np.float)
                spid2pixnum[segments[y, x]] = spid2pixnum.get(segments[y, x], 0) + 1

        x_test = list()
        x_test_coord = list()

        spid2featureid = dict()
        for spid in spid2center.keys():  # spid2center.keys()多少个超像素块
            spid2center[spid] = spid2center[spid] / float(spid2pixnum[spid])  # 超像素块的坐标和/超像素块中像素点的个数
            x_test_coord.append([spid2center[spid][0] / float(h - 1), spid2center[spid][1] / float(w - 1)])  # 相对坐标
            spid2featureid[spid] = len(x_test)

            # ==================================================================
            # 根据分割的结果得到81个随机的像素值，形成9*9的patch
            patch = list()
            coord = np.where(segments == spid)
            index = np.random.randint(len(coord[0]), size=81)
            a = (zip(coord[0][index], coord[1][index]))
            for i, j in (zip(coord[0][index], coord[1][index])):  # i:x坐标，j:y坐标
                patch.append(img[i,j]/255.)
            patch = np.array(patch)
            x_test.append(patch.reshape(9,9,3))
        # print("spid2featureid:",spid2featureid)

        # 2. recolor
        # (1) pretrain
        t0 = time.time()
        Y_label = np_utils.to_categorical(y_train, 3)  # 这里的输出类别也要改！！！
        # print("Pretrain...")
        self.dnn.train(np.asarray(x_train), np.asarray(Y_label))
        # print("Done...")
        # print("Pretraining Time:", time.time() - t0)

        # # (2) train
        # print("Finetune...")
        # for i in range(1):
        #     model = finetune()
        #     # 终止条件2；终止条件1还没实现!!!#编写自己的回调函数
        #     # 2017.5.22迭代一期，进行一次测试集的输出，并加上相应的损失
        #     ES = EarlyStopping(monitor='loss', min_delta=0.01, patience=0, verbose=1, mode='auto')
        #     history = model.fit([np.asarray(x_train), np.asarray(x_train_coord)], np.asarray(Y_label),
        #                         verbose=0, batch_size=10,
        #                         epochs=20,
        #                         callbacks=[ES]
        #                         )
        #     model.save_weights('my_model.h5')
        # print("Done...")
        # print("Training Time:", time.time() - t0)

        # (3) estimate
        t0 = time.time()

        print("Estimate...")
        # model = finetune()
        # model.load_weights("my_model.h5")

        Y_label = self.dnn.estimate([np.asarray(x_test), np.asarray(x_test_coord)], batch_size=100, verbose=0)
        print("Done...")
        print("Estimation Time:", time.time() - t0)

        # print("segments:", segments)
        # print("segments[0,0]:", segments[0,0])

        # (4) colorize
        c_img = np.zeros((h, w, 3), np.uint8)
        for y in range(h):
            for x in range(w):
                spid = segments[y, x]
                featureid = spid2featureid[spid]
                probs = Y_label[featureid]
                res_color = np.zeros(3, np.float)

                for label, prob in enumerate(probs):  # probs:类似[ 0.28122437  0.13114531  0.58763027]
                    color = label2color[label]  # 得到颜色:类似array([  64.,  128.,  255.])
                    if color.dot(color) == 0.:  # 颜色:array([  64.,  128.,  255.]),得到:64*64+128*128+255*255
                        color = img[y, x]  # 等于0，意味着为背景色！不变
                    res_color += color * prob  # 三类标签颜色的叠加

                c_img[y, x] = np.uint8(res_color)


        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2Lab)  # 将染色后的c_img转到Lab颜色空间
        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # 将原图转到Lab颜色空间
        res_img = np.c_[img_Lab[:, :, :1], c_img[:, :, 1].reshape(h, w, 1), c_img[:, :, 2].reshape(h, w, 1)]
        res_img = cv2.cvtColor(res_img, cv2.COLOR_Lab2BGR)


        # 3. postprocess
        U_list = self.postprocessing(img, [c_img[:, :, 1], c_img[:, :, 2]], 10, 0.01)
        res_img = np.c_[img_Lab[:, :, :1], U_list[0], U_list[1]]
        res_img = cv2.cvtColor(res_img, cv2.COLOR_Lab2BGR)

        cv2.imwrite('data/sample_sheep_out.jpg', res_img)

    def postprocessing(self, img, Z_list, lmbd, sigma):
        h, w = img.shape[:2]
        edge_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200, L2gradient=True)
        edge_img = cv2.dilate(edge_img, np.ones((2, 2), np.uint8), iterations=2)
        normalized_img = np.copy(img) / 255.

        b_list = list()
        for Z in Z_list:
            b_list.append(np.array(Z.reshape(h * w), np.float))

        data = list()
        row_id = list()
        col_id = list()
        for y in range(0, h):
            for x in range(0, w):
                yw = y * w
                yw_x = yw + x
                val = 0.
                if y + 1 < h:
                    c_vec = normalized_img[y + 1, x] - normalized_img[y, x]
                    weight = lmbd * np.exp(-c_vec.dot(c_vec) / sigma)
                    row_id.append(yw_x)
                    col_id.append(yw_x + w)
                    data.append(-weight)
                    val += weight
                if y - 1 >= 0:
                    c_vec = normalized_img[y - 1, x] - normalized_img[y, x]
                    weight = lmbd * np.exp(-c_vec.dot(c_vec) / sigma)
                    row_id.append(yw_x)
                    col_id.append(yw_x - w)
                    data.append(-weight)
                    val += weight
                if x + 1 < w:
                    c_vec = normalized_img[y, x + 1] - normalized_img[y, x]
                    weight = lmbd * np.exp(-c_vec.dot(c_vec) / sigma)
                    row_id.append(yw_x)
                    col_id.append(yw_x + 1)
                    data.append(-weight)
                    val += weight
                if x - 1 >= 0:
                    c_vec = normalized_img[y, x - 1] - normalized_img[y, x]
                    weight = lmbd * np.exp(-c_vec.dot(c_vec) / sigma)
                    row_id.append(yw_x)
                    col_id.append(yw_x - 1)
                    data.append(-weight)
                    val += weight

                row_id.append(yw_x)
                col_id.append(yw_x)
                if edge_img[y, x] == 255:
                    data.append(val)
                    for b in b_list:
                        b[yw_x] = 0.
                else:
                    data.append(val + 1.)

        A = scipy.sparse.coo_matrix((data, (row_id, col_id)), shape=(h * w, h * w))
        A = scipy.sparse.csc_matrix(A)

        U_list = list()
        for b in b_list:
            lu = scipy.sparse.linalg.splu(A)
            U = lu.solve(b)
            U = np.uint8(U.reshape(h, w, 1))
            U_list.append(U)

        return U_list

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', help='file path of input image',
                        default='../data/flower.jpg')
    parser.add_argument('-s', help='file path of user stroke', default='../data/strk.png')
    parser.add_argument('-o', help='file path of output image', default='data/out.jpg')
    parser.add_argument('-gpu', help='GPU device specifier', default='-1')
    args = parser.parse_args()

    DP = DeepProp(img_path=args.i)
    t0 = time.time()
    DP.recoloring(strk_path=args.s, out_path=args.o)
    print("Total Time:", time.time() - t0)


if __name__ == '__main__':
    main()