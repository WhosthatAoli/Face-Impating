import cv2
import dlib
import numpy
import sys
import matplotlib.pyplot as plt
import os

PREDICTOR_PATH = "align_model/shape_predictor_68_face_landmarks.dat"  # 68个关键点landmarks的模型文件
SCALE_FACTOR = 1  # 图像的放缩比
FEATHER_AMOUNT = 15  # 羽化边界范围，越大，羽化能力越大，一定要奇数，不能偶数

# 　68个点
FACE_POINTS = list(range(17, 68))  # 脸
MOUTH_POINTS = list(range(48, 61))  # 嘴巴
RIGHT_BROW_POINTS = list(range(17, 22))  # 右眉毛
LEFT_BROW_POINTS = list(range(22, 27))  # 左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))  # 右眼睛
LEFT_EYE_POINTS = list(range(42, 48))  # 左眼睛
NOSE_POINTS = list(range(27, 35))  # 鼻子
JAW_POINTS = list(range(0, 17))  # 下巴

# 选取用于叠加在第一张脸上的第二张脸的面部特征
# 特征点包括左右眼、眉毛、鼻子和嘴巴

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
# 眼睛 ，眉毛             2 * 22
# 鼻子，嘴巴   分开来

# 定义用于颜色校正的模糊量，作为瞳孔距离的系数
COLOUR_CORRECT_BLUR_FRAC = 0.6

# 实例化脸部检测器
detector = dlib.get_frontal_face_detector()
# 加载训练模型
# 并实例化特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# 定义了两个类处理意外
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    '''
    通过predictor 拿到68 landmarks
    '''
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])  # 68*2的矩阵


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


def warp_im(im, M, dshape):
    '''
    由 get_face_mask 获得的图像掩码还不能直接使用，因为一般来讲用户提供的两张图像的分辨率大小很可能不一样，而且即便分辨率一样，
    图像中的人脸由于拍摄角度和距离等原因也会呈现出不同的大小以及角度，所以如果不能只是简单地把第二个人的面部特征抠下来直接放在第一个人脸上，
    我们还需要根据两者计算所得的面部特征区域进行匹配变换，使得二者的面部特征尽可能重合。

    仿射函数，warpAffine，能对图像进行几何变换
        三个主要参数，第一个输入图像，第二个变换矩阵 np.float32 类型，第三个变换之后图像的宽高

    对齐主要函数
    '''
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


# 人脸对齐函数
def face_Align(Base_path, cover_path):
    im1, landmarks1 = read_im_and_landmarks(Base_path)  # 底图
    im2, landmarks2 = read_im_and_landmarks(cover_path)  # 贴上来的图

    if len(landmarks1) == 0 & len(landmarks2) == 0:
        print("Faces detected is no face!")
    if len(landmarks1) > 1 & len(landmarks2) > 1:
        print("Faces detected is more than 1!")

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    warped_im2 = warp_im(im2, M, im1.shape)
    return warped_im2

def preprocess(filename):
    '''
    人脸对齐-dlib
    Base_path:模板图
    cover_path：需对齐的图
    '''
    Base_path = './test/image/basic.jpg'
    cover_path = filename
    face_aligned = face_Align(Base_path, cover_path)
    cv2.imwrite(filename, face_aligned)
