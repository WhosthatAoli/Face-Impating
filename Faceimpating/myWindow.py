from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene, QMessageBox, QSpinBox, QLabel
from PyQt5.QtGui import QImage, QPixmap, QMovie
from PIL import Image
from gui import Ui_MainWindow
from PaintBoard import PaintBoard
from main import main
from face_to_face import preprocess

import numpy as np
import random
import cv2
import dlib
import os


class myWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(myWindow, self).__init__()
        self.setupUi(self)

        # 画板
        self.paint_board = PaintBoard(self)
        self.paint_board.move(20, 85)
        self.paint_board.setEnabled(False)

        # 设置Enable
        self.menu.setEnabled(True)
        self.menuMask.setEnabled(False)
        self.menuRandom_Mask.setEnabled(False)
        self.menuFree_Mask.setEnabled(False)
        self.pushButton_Compare.setEnabled(False)
        self.pushButton_restore.setEnabled(False)
        self.pushButton_gen.setEnabled(False)

        # 绑定槽函数
        self.actionOpen.triggered.connect(self.openImage)
        self.actionSave.triggered.connect(self.saveImage)

        self.actionCentral_Mask.triggered.connect(lambda: self.addMask("central"))
        self.actionTop_Left_Corner.triggered.connect(lambda: self.addMask("lu"))
        self.actionBottom_Left_Corner.triggered.connect(lambda: self.addMask("ld"))
        self.actionTop_Right_Corner.triggered.connect(lambda: self.addMask("ru"))
        self.actionBottom_Right_Corner.triggered.connect(lambda: self.addMask("rd"))
        self.actionLeft.triggered.connect(lambda: self.addMask("left"))
        self.actionRight.triggered.connect(lambda: self.addMask("right"))
        self.actionTop.triggered.connect(lambda: self.addMask("up"))
        self.actionBottom.triggered.connect(lambda: self.addMask("down"))
        self.actionRandom_Block.triggered.connect(lambda: self.addMask("block"))
        self.actionRandom_Walk.triggered.connect(lambda: self.addMask("walk"))
        self.actionwearMask.triggered.connect(self.show_morpher_pic)

        self.menuFree_Mask.triggered.connect(self.draw)
        self.actionThickness.triggered.connect(self.changeThickness)

        self.pushButton_restore.clicked.connect(self.restore)
        self.pushButton_gen.clicked.connect(self.generate)
        self.pushButton_Compare.pressed.connect(lambda: self.showCompare("press"))
        self.pushButton_Compare.released.connect(lambda: self.showCompare("release"))

    def openImage(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, 'open', r"./")
        img = QImage()
        img.load(self.filename)
        if img.width() != 256:
            preprocess(self.filename)
        img.load(self.filename)
        self.width = img.width()
        self.height = img.height()
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap().fromImage(img))
        self.graphicsView_img.setScene(scene)

        self.menuMask.setEnabled(True)
        self.menuRandom_Mask.setEnabled(True)
        self.menuFree_Mask.setEnabled(True)

    def restore(self):
        self.paint_board.Clear()
        img = QImage()
        img.load(self.filename)
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap().fromImage(img))
        self.graphicsView_img.setScene(scene)

        self.pushButton_gen.setEnabled(False)

    def create_mask(self, width, height, mask_width, mask_height, x=None, y=None):
        mask = np.zeros((height, width))
        mask_x = x if x is not None else random.randint(0, width - mask_width)
        mask_y = y if y is not None else random.randint(0, height - mask_height)
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    def random_walk(self, width, height):
        mask = np.zeros((height, width))
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        x_list = []
        y_list = []
        action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for i in range(width ** 2):
            r = random.randint(0, len(action_list) - 1)
            x = np.clip(x + action_list[r][0], a_min=0, a_max=width - 1)
            y = np.clip(y + action_list[r][1], a_min=0, a_max=height - 1)
            x_list.append(x)
            y_list.append(y)
        mask[np.array(x_list), np.array(y_list)] = 1
        return mask

    def addMask(self, type):
        # 清空面板上的掩码
        self.restore()
        if type == "central":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height // 2, self.width // 4,
                                         self.height // 4)
        elif type == "lu":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height // 2, 0, 0)
        elif type == "ld":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height // 2, 0,
                                         self.height // 2)
        elif type == "ru":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height // 2, self.width // 2, 0)
        elif type == "rd":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height // 2, self.width // 2,
                                         self.height // 2)
        elif type == "left":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height, 0, 0)
        elif type == "right":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height, self.width // 2, 0)
        elif type == "up":
            self.mask = self.create_mask(self.width, self.height, self.width, self.height // 2, 0, 0)
        elif type == "down":
            self.mask = self.create_mask(self.width, self.height, self.width, self.height // 2, 0, self.height // 2)
        elif type == "block":
            self.mask = self.create_mask(self.width, self.height, self.width // 2, self.height // 2)
        elif type == "walk":
            self.mask = self.random_walk(self.width, self.height)

        self.mask = self.mask * 255
        self.mask = Image.fromarray(np.uint8(self.mask))
        self.maskfile = './test/mask/mask.jpg'
        self.mask.save(self.maskfile)
        # 为图片加上掩码
        img = cv2.imread(self.filename)
        mask = cv2.imread(self.maskfile, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        mask = 255 - mask
        masked = cv2.add(img, np.ones(np.shape(img), dtype=np.uint8), mask=mask)
        self.maskedfile = './test/image_masked/masked.jpg'
        cv2.imwrite(self.maskedfile, masked)
        # 显示图片
        img = QImage()
        img.load(self.maskedfile)
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap().fromImage(img))
        self.graphicsView_img.setScene(scene)

        self.pushButton_gen.setEnabled(True)
        self.pushButton_restore.setEnabled(True)

    # 人脸戴口罩效果展示
    def show_morpher_pic(self):
        print(self.filename)
        img1 = cv2.imread(self.filename)
        x_min, x_max, y_min, y_max, size = self.get_mouth(img1)
        print("find mouse")
        self.mask = Image.open('./test/mask.png')
        adding = self.mask.resize(size)
        im = Image.fromarray(img1[:, :, ::-1])  # 切换RGB格式
        # 在合适位置添加头发图片
        im.paste(adding, (int(x_min), int(y_min)), adding)
        # im.show()
        self.maskedfile = './test/image_masked/masked.jpg'
        print(self.maskedfile)
        im.save(self.maskedfile)

        # 制作mask
        bg_path = './test/bg.jpg'
        mask = cv2.imread(bg_path)
        mask = Image.fromarray(mask[:, :, ::-1])  # 切换RGB格式
        # 添加mask
        mask.paste(adding, (int(x_min), int(y_min)), adding)

        # file_in = './test/mask/mask.jpg'
        # file_out = './test/mask/mask_new.jpg'
        # image = Image.open(file_in)
        mask = mask.convert('L')

        #  setup a converting table with constant threshold
        threshold = 20
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)

        # convert to binary image by the table
        mask = mask.point(table, '1')
        self.maskfile = './test/mask/mask.jpg'
        mask.save(self.maskfile)

        # 显示图片
        img = QImage()
        img.load(self.maskedfile)
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap().fromImage(img))
        self.graphicsView_img.setScene(scene)

        self.pushButton_gen.setEnabled(True)
        self.pushButton_restore.setEnabled(True)

    def get_mouth(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./align_model/shape_predictor_68_face_landmarks.dat')
        faces = detector(img_gray, 0)
        for k, d in enumerate(faces):
            x = []
            y = []
            # 人脸大小的高度
            height = d.bottom() - d.top()
            # 人脸大小的宽度
            width = d.right() - d.left()
            shape = predictor(img_gray, d)
            # 48-67 为嘴唇部分
            for i in range(48, 68):
                x.append(shape.part(i).x)
                y.append(shape.part(i).y)
            # 根据人脸的大小扩大嘴唇对应口罩的区域
            y_max = (int)(max(y) + height / 3.2)
            y_min = (int)(min(y) - height / 3.2)
            x_max = (int)(max(x) + width / 3.2)
            x_min = (int)(min(x) - width / 3.2)
            size = ((x_max - x_min), (y_max - y_min))
            return x_min, x_max, y_min, y_max, size

    def generate(self):
        print(1)
        main(mode=2, input=self.filename, mask=self.maskfile)
        print(1)
        name = self.filename.split('/')[-1]
        print(name)
        name = name[:-4]+'.png'
        self.result = os.path.join('./checkpoints/results/landmark_inpaint/result', name)

        # 显示图片
        img = QImage()
        img.load(self.result)
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap().fromImage(img))
        self.graphicsView_result.setScene(scene)

        self.pushButton_Compare.setEnabled(True)

    def saveImage(self):
        # 获取文件路径
        file_path = self.result
        print(file_path)
        file_name = QFileDialog.getSaveFileName(self, "文件保存", "D:/imagetest/save",
                                                "All Files (*);;Image Files (*.png);;Image Files (*.jpg)")
        print(file_name[0])
        img = Image.open(file_path)
        img.save(file_name[0])

    def showCompare(self, type):
        showfile = ""
        if type == "press":
            showfile = self.filename
        elif type == "release":
            showfile = self.result
        # 显示图片
        img = QImage()
        img.load(showfile)
        scene = QGraphicsScene()
        scene.addPixmap(QPixmap().fromImage(img))
        self.graphicsView_result.setScene(scene)

    def draw(self):
        print("Draw")
        self.restore()
        self.paint_board.setEnabled(True)
        self.maskfile = './test/mask/mask.jpg'
        self.pushButton_restore.setEnabled(True)
        self.pushButton_gen.setEnabled(True)

    def changeThickness(self):
        QMessageBox.information(self, "设置画笔大小", "画笔大小为", QMessageBox.Ok)
        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(20)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(10)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
