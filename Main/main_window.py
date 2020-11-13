import os
import sys
import time
import cv2
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from Main.MainWindow import Ui_MainWindow
from Hough.hough_cricles import HoughCricles
from Binarization.binarization import AreaBinarization


class Main(Ui_MainWindow, QMainWindow):
    # 连接子窗口
    def __init__(self):
        super().__init__()
        # 原图:用于处理
        self.picture_cv_obj = None
        # 原图:用于历史图
        self.picture2_cv_obj = None
        # 当前图
        self.img_cv_obj = None
        # 存放临时数据
        self.data = None
        # 图片列表:用于历史图
        self.img_list = list()
        # 存储当前时间
        self.click_time = time.time()
        # 记录打开图片的路径
        self.open_img_str = str()
        # 记录打开图片所在的文件夹
        self.open_dir_str = str()
        # 设置历史区域存储图片的数量
        self.tiny_img_number = 10
        # 引入GUI的setupUi
        self.setupUi(self)
        # 重写起始坐标
        pass
        # 双击历史图
        self.pushButton_00.clicked.connect(lambda: self.read_historical_img(0))
        self.pushButton_01.clicked.connect(lambda: self.read_historical_img(1))
        self.pushButton_02.clicked.connect(lambda: self.read_historical_img(2))
        self.pushButton_03.clicked.connect(lambda: self.read_historical_img(3))
        self.pushButton_04.clicked.connect(lambda: self.read_historical_img(4))
        self.pushButton_05.clicked.connect(lambda: self.read_historical_img(5))
        self.pushButton_06.clicked.connect(lambda: self.read_historical_img(6))
        self.pushButton_07.clicked.connect(lambda: self.read_historical_img(7))
        self.pushButton_08.clicked.connect(lambda: self.read_historical_img(8))
        self.pushButton_09.clicked.connect(lambda: self.read_historical_img(9))
        # 单机功能
        self.pushButton_10.clicked.connect(self.open_picture)  # 打开
        self.pushButton_11.clicked.connect(self.save_img)  # 保存
        self.pushButton_12.clicked.connect(self.show_picture)  # 显示原图
        self.pushButton_13.clicked.connect(self.change_img2picture)  # 置入
        self.pushButton_14.clicked.connect(self.flip_img)  # 图像翻转
        self.pushButton_15.clicked.connect(self.adaptiveThreshold)  # 自适应二值化
        self.pushButton_16.clicked.connect(self.threshold)  # 二值化
        self.pushButton_17.clicked.connect(self.erode_img)  # 腐蚀
        self.pushButton_18.clicked.connect(self.dilate_img)  # 膨胀
        self.pushButton_19.clicked.connect(self.morph_open)  # 开运算
        self.pushButton_20.clicked.connect(self.morph_close)  # 闭运算
        self.pushButton_21.clicked.connect(self.morph_gradient)  # 形态学梯度运算
        self.pushButton_22.clicked.connect(self.morph_tophat)  # 顶帽运算
        self.pushButton_23.clicked.connect(self.morph_blackhat)  # 黑帽运算
        self.pushButton_24.clicked.connect(lambda: self.rotate_img(0))  # 旋转
        self.pushButton_25.clicked.connect(self.resize_img)  # 缩放
        self.pushButton_26.clicked.connect(self.clear)  # 清空
        self.pushButton_27.clicked.connect(self.bgr_equalize_hist)  # 彩色直方图
        self.pushButton_28.clicked.connect(self.gray_equalize_hist)  # 灰度直方图
        self.pushButton_29.clicked.connect(self.inv_threshold)  # 阈值翻转
        self.pushButton_30.clicked.connect(self.show_difference)  # 差异
        self.pushButton_31.clicked.connect(self.sharpen_img)  # 锐化
        self.pushButton_32.clicked.connect(self.blur_img)  # 模糊
        # 数字调节框
        self.doubleSpinBox_00.valueChanged.connect(self.threshold)
        self.doubleSpinBox_01.valueChanged.connect(self.erode_img)
        self.doubleSpinBox_02.valueChanged.connect(self.dilate_img)
        self.doubleSpinBox_03.valueChanged.connect(self.morph_open)
        self.doubleSpinBox_04.valueChanged.connect(self.morph_close)
        self.doubleSpinBox_05.valueChanged.connect(self.morph_gradient)
        self.doubleSpinBox_06.valueChanged.connect(self.morph_tophat)
        self.doubleSpinBox_07.valueChanged.connect(self.morph_blackhat)
        self.doubleSpinBox_08.valueChanged.connect(lambda: self.rotate_img(1))
        self.doubleSpinBox_09.valueChanged.connect(self.resize_img)  # 缩放
        self.doubleSpinBox_10.valueChanged.connect(self.blur_img)  # 模糊
        # 单选按钮
        self.radioButton.clicked.connect(lambda: self.show_layer('bgr'))
        self.radioButton_00.clicked.connect(lambda: self.show_layer('G'))
        self.radioButton_01.clicked.connect(lambda: self.show_layer('b'))
        self.radioButton_02.clicked.connect(lambda: self.show_layer('g'))
        self.radioButton_03.clicked.connect(lambda: self.show_layer('r'))
        self.radioButton_04.clicked.connect(lambda: self.show_layer('h'))
        self.radioButton_05.clicked.connect(lambda: self.show_layer('s'))
        self.radioButton_06.clicked.connect(lambda: self.show_layer('v'))
        self.radioButton_07.clicked.connect(lambda: self.show_layer('L'))
        self.radioButton_08.clicked.connect(lambda: self.show_layer('A'))
        self.radioButton_09.clicked.connect(lambda: self.show_layer('B'))
        self.radioButton_10.clicked.connect(lambda: self.show_layer('all'))
        # 下拉列表
        self.comboBox_1.currentIndexChanged.connect(self.flip_img)  # 图像翻转
        self.comboBox_2.currentIndexChanged.connect(self.adaptiveThreshold)  # 自适应二值化
        self.comboBox_3.currentIndexChanged.connect(self.sharpen_img)  # 锐化
        self.comboBox_4.currentIndexChanged.connect(self.blur_img)  # 模糊
        # 二级窗口
        self.pushButton_33.clicked.connect(self.hough_circles)  # 霍夫圆检测
        # self.pushButton_34.clicked.connect()  # 霍夫直线变换
        self.pushButton_35.clicked.connect(self.area_binarization)  # 自定义区域二值化
        # 打印参数
        self.label_0.setText('打印参数')
        # 重写ico路径
        icon = QtGui.QIcon()
        ico_path = './main.ico' if __name__ == '__main__' else './Main/main.ico'
        icon.addPixmap(QtGui.QPixmap(ico_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        # 读取封面图
        self.show_cover()

    # 显示封面
    def show_cover(self):
        try:
            self.open_img_str = './cover.png' if __name__ == '__main__' else 'Main/cover.png'
            self.open_dir_str = os.path.dirname(self.open_img_str)
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj = cv2.imread(self.open_img_str)
            self.add_img_list(self.img_cv_obj)
            self.show_img()
            self.show_tiny_img()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '封面不存在!', str(e))
            msg_box.exec_()

    def whether_open_img(funciton):
        """检查是否已打开文件的装饰器"""

        # noinspection PyCallingNonCallable
        def func(self):
            if self.open_img_str:
                funciton(self)
            else:
                self.open_picture()
                funciton(self)

        return func

    def open_picture(self):
        img_path, img_type = QFileDialog.getOpenFileName(self, "打开图片", self.open_dir_str,
                                                         "All Files(*.bmp *.png *.jpg *.jpeg)")
        if not img_path:
            return
        try:
            self.open_img_str = img_path
            self.open_dir_str = os.path.dirname(self.open_img_str)
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj = cv2.imread(self.open_img_str)
            self.add_img_list(self.img_cv_obj)
            self.show_img()
            self.show_tiny_img()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片打开异常!', str(e))
            msg_box.exec_()

    def add_img_list(self, img):
        try:
            self.img_list.insert(0, img)
            if len(self.img_list) > self.tiny_img_number:
                self.img_list.pop()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '保存历史异常!', str(e))
            msg_box.exec_()

    def show_img(self):
        try:
            qt_img_obj = self.cv2qt(self.img_cv_obj)
            pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pixmap)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片显示异常!', str(e))
            msg_box.exec_()

    def show_tiny_img(self):
        try:
            for i, img in enumerate(self.img_list):
                qt_img_obj = self.cv2qt(img)
                label_name = eval('self.label_0' + str(i))
                pixmap = QPixmap.fromImage(qt_img_obj).scaled(label_name.width(), label_name.height())
                label_name.setPixmap(pixmap)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '历史图显示异常!', str(e))
            msg_box.exec_()

    def change_img2picture(self):
        """置入图片"""
        try:
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj
            self.show_picture()
            self.add_img_list(self.img_cv_obj)
            self.show_tiny_img()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片置入异常!', str(e))
            msg_box.exec_()

    @staticmethod
    def cv2qt(img) -> QImage:
        """图片类型转换"""
        try:
            image_height, image_width, image_depth = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, image_width, image_height, image_width * image_depth, QImage.Format_RGB888)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '类型转换异常!', str(e))
            msg_box.exec_()

    def is_double_click(self) -> bool:
        """事件:判断是否双击"""
        t = time.time()
        delta = t - self.click_time
        self.click_time = t
        if delta < 0.3:
            return True

    def read_historical_img(self, value):
        """读取历史图"""
        if not self.is_double_click():
            return
        if len(self.img_list) > value:
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj = self.img_list.pop(value)
            self.add_img_list(self.img_cv_obj)
            self.show_img()
            self.show_tiny_img()

    def get_image_checkbox(self) -> cv2:
        """根据复选框状态(是否叠加)获取图片"""
        if self.checkBox.isChecked():
            return self.img_cv_obj
        else:
            return self.picture_cv_obj

    def clear(self):
        """清除图片"""
        self.label.clear()
        for i in range(len(self.img_list)):
            label_name = eval('self.label_0' + str(i))
            label_name.clear()
        self.picture_cv_obj = None
        self.picture2_cv_obj = None
        self.img_cv_obj = None
        self.img_list = list()
        self.click_time = time.time()
        self.open_img_str = str()
        self.open_dir_str = str()

    @whether_open_img
    def show_picture(self):
        """显示原图"""
        try:
            qt_img_obj = self.cv2qt(self.picture_cv_obj)
            pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pixmap)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '原图显示异常!', str(e))
            msg_box.exec_()

    @whether_open_img
    def save_img(self):
        """保存图片"""
        try:
            img_name, img_type = QFileDialog.getSaveFileName(None, "保存图片", self.open_dir_str, "*.jpg;;*.png")
            if img_name:
                cv2.imwrite(img_name, self.img_cv_obj)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片保存异常!', str(e))
            msg_box.exec_()

    def show_layer(self, letter):
        """显示图层"""
        if not self.open_img_str:
            return self.open_picture()
        if letter == 'bgr':
            self.picture_cv_obj = self.img_cv_obj = self.picture2_cv_obj
            return self.show_img()
        bgr = self.picture2_cv_obj
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(bgr)
        h, s, v = cv2.split(hsv)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        G = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if letter == 'all':
            all = cv2.vconcat((cv2.hconcat((b, g, r)), cv2.hconcat((h, s, v)), cv2.hconcat((L, A, B))))
            all = cv2.resize(all, bgr.shape[0:2])
        img = cv2.cvtColor(eval(letter), cv2.COLOR_GRAY2BGR)
        self.picture_cv_obj = self.img_cv_obj = img
        qt_img_obj = self.cv2qt(self.img_cv_obj)
        pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(pixmap)

    @whether_open_img
    def flip_img(self):
        """翻转"""
        img = self.get_image_checkbox()
        index = self.comboBox_1.currentIndex()
        self.img_cv_obj = cv2.flip(img, index)
        self.show_img()

    @whether_open_img
    def inv_threshold(self):
        """翻转二值化"""
        img = self.get_image_checkbox()
        ret, self.img_cv_obj = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        self.show_img()

    @whether_open_img
    def adaptiveThreshold(self):
        """自适应二值化"""
        img = self.get_image_checkbox()
        index = self.comboBox_2.currentIndex()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if index == 0:
            adaptive_img = cv2.adaptiveThreshold(gray, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY, 11, 2)
        elif index == 1:
            adaptive_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif index == 2:
            adaptive_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # noinspection PyUnboundLocalVariable
        self.img_cv_obj = cv2.cvtColor(adaptive_img, cv2.COLOR_GRAY2BGR)
        self.show_img()

    @whether_open_img
    def threshold(self):
        """二值化"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_00.value())
        ret, self.img_cv_obj = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
        self.show_img()

    @whether_open_img
    def erode_img(self):
        """腐蚀"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_01.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.erode(img, kernel)
        self.show_img()

    @whether_open_img
    def dilate_img(self):
        """膨胀"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_02.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.dilate(img, kernel)
        self.show_img()

    @whether_open_img
    def morph_open(self):
        """开运算"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_03.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        self.show_img()

    @whether_open_img
    def morph_close(self):
        """闭运算"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_04.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        self.show_img()

    @whether_open_img
    def morph_gradient(self):
        """形态学梯度运算"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_05.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        self.show_img()

    @whether_open_img
    def morph_tophat(self):
        """顶帽运算"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_06.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        self.show_img()

    @whether_open_img
    def morph_blackhat(self):
        """黑帽运算"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_07.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        self.show_img()

    def rotate_img(self, value):
        """ 旋转
        :param value: click事件为0 ,change事件为1
        :return:
        """
        # 选择 & 不点击时直接返回 不操作!
        if not self.open_img_str:
            return
        if self.checkBox.isChecked() and value == 1:
            return
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_08.value())
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        # 计算旋转矩阵:(中心,角度,缩放比)
        m = cv2.getRotationMatrix2D(center, val, 1)
        # 使用openCV仿射变换实现函数旋转
        self.img_cv_obj = cv2.warpAffine(img, m, (w, h))
        self.show_img()

    @whether_open_img
    def resize_img(self):
        """缩放"""
        img = self.get_image_checkbox()
        val = self.doubleSpinBox_09.value()
        h, w = img.shape[:2]
        w, h = int(w * val), int(h * val)
        print(w, h)
        self.img_cv_obj = cv2.resize(img, (w, h))
        self.show_img()

    @whether_open_img
    def blur_img(self):
        """模糊"""
        index = self.comboBox_4.currentIndex()
        val = int(self.doubleSpinBox_10.value())
        img = self.get_image_checkbox()
        if index == 0:
            self.img_cv_obj = cv2.medianBlur(img, val)
        elif index == 1:
            self.img_cv_obj = cv2.GaussianBlur(img, (val, val), 3)
        elif index == 2:
            self.img_cv_obj = cv2.blur(img, (val, val))
        self.show_img()

    @whether_open_img
    def sharpen_img(self):
        """锐化"""
        img = self.get_image_checkbox()
        index = self.comboBox_3.currentIndex()
        # 锐化算子1
        sharpen_0 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # 锐化算子2
        sharpen_1 = np.array([[0, -1, 0], [-1, 8, -1], [0, 1, 0]]) / 4.0
        # 选择算子
        sharpen_operator = eval('sharpen_' + str(index))
        # 使用filter2D进行滤波操作
        self.img_cv_obj = cv2.filter2D(img, -1, sharpen_operator)
        self.show_img()

    @whether_open_img
    def bgr_equalize_hist(self):
        """彩色直方图均衡化"""
        img = self.get_image_checkbox()
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
        self.img_cv_obj = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.show_img()

    @whether_open_img
    def gray_equalize_hist(self):
        img = self.get_image_checkbox()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.equalizeHist(gray)
        self.img_cv_obj = cv2.cvtColor(gray_hist, cv2.COLOR_GRAY2BGR)
        self.show_img()

    @whether_open_img
    def show_difference(self):
        """显示差异"""
        if len(self.img_list) < 2:
            return self.open_picture()
        a = self.img_list[0]
        b = self.img_list[1]
        self.img_cv_obj = cv2.subtract(a, b)
        self.show_img()

    # todo 跳转
    @whether_open_img
    def hough_circles(self):
        """霍夫圆变换"""
        self.hide()
        HcWin = HoughCricles(self, self.picture2_cv_obj, self.img_cv_obj)
        HcWin.exec()
        if self.data:
            for item in self.data:
                self.img_cv_obj = item
                self.change_img2picture()
        self.data = None
        self.show()

    @whether_open_img
    def area_binarization(self):
        """自定义区域二值化 """
        self.hide()
        AbWin = AreaBinarization(self, self.picture2_cv_obj, self.img_cv_obj)
        AbWin.exec()
        if self.data:
            for item in self.data:
                self.img_cv_obj = item
                self.change_img2picture()
        self.data = None
        self.show()


def run():
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
