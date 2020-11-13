import cv2
import copy
from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from Hough.GetCriclesWindow import Ui_Form


class HoughCricles(Ui_Form, QDialog):
    def __init__(self, mainwindow, picture, image):
        super(HoughCricles, self).__init__()

        self.mainwindow = mainwindow
        # 原图:传入用于备份
        self.picture_cv_obj = picture
        # 主图:由当前图转灰度而成,用于实际的操作变换
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.main_img_cv_obj = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 缓存彩图:在原图基础上画添加辅助线
        self.bgr_cv_obj = None
        # 缓存灰度图:
        self.gray_cv_obj = None

        # 引入GUI的setupUi
        self.setupUi(self)
        # 重设ico图标
        icon = QtGui.QIcon()
        ico_path = "./Hough/logo.ico"
        icon.addPixmap(QtGui.QPixmap(ico_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        # # 重写:图例背景图
        self.label_00.setPixmap(QtGui.QPixmap("./Hough/legend.jpg"))
        # 点击事件
        self.pushButton_1.clicked.connect(self.reload)  # 重载
        self.pushButton_2.clicked.connect(self.return_img)  # 返回
        self.pushButton_3.clicked.connect(self.show_img)  # 原图
        self.pushButton_4.clicked.connect(self.operation)  # 开始变换
        # 双精度数字调节框
        self.doubleSpinBox_01.valueChanged.connect(self.operation)  # 圆心距
        self.doubleSpinBox_02.valueChanged.connect(self.operation)  # 圆数量
        # self.doubleSpinBox_03.valueChanged.connect(self.operation)  # 二值化
        self.doubleSpinBox_03.valueChanged.connect(self.operation)  # 高斯模糊
        self.doubleSpinBox_04.valueChanged.connect(self.operation)  # 最大半径
        self.doubleSpinBox_05.valueChanged.connect(self.operation)  # 最小半径
        self.doubleSpinBox_07.valueChanged.connect(self.operation)  # 内圆半径
        self.doubleSpinBox_08.valueChanged.connect(self.operation)  # 外圆半径
        self.doubleSpinBox_09.valueChanged.connect(self.operation)  # 向内修正
        self.doubleSpinBox_10.valueChanged.connect(self.operation)  # 向外修正
        self.doubleSpinBox_11.valueChanged.connect(self.operation)  # 1区阈值
        self.doubleSpinBox_12.valueChanged.connect(self.operation)  # 2区阈值
        self.doubleSpinBox_13.valueChanged.connect(self.operation)  # 3区阈值
        self.doubleSpinBox_14.valueChanged.connect(self.operation)  # 4区阈值
        # 显示图片
        self.reload()

    # 重载
    def reload(self):
        # 缓存彩图
        self.bgr_cv_obj = copy.deepcopy(self.picture_cv_obj)
        # 缓存灰度图:
        self.gray_cv_obj = copy.deepcopy(self.main_img_cv_obj)
        self.show_img()

    def return_img(self):
        self.mainwindow.data = self.bgr_cv_obj, self.gray_cv_obj
        self.close()

    # 将图片显示到标签上
    def show_img(self):
        # 根据按钮的选中状态赋值
        qt_img_obj = self.cv2qt(self.bgr_cv_obj) if self.pushButton_3.isChecked() else self.cv2qt(self.gray_cv_obj)

        try:
            pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pixmap)

        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片显示异常!', str(e) + '\n GetArea.py 40 line')
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



    # todo 功能 ########################################################################################################
    def operation(self):
        # 参数获取:
        min_dist = int(self.doubleSpinBox_01.value())  # 最小圆心距
        show_mode = int(self.doubleSpinBox_02.value())  # 显示圆数量
        blur_val = int(self.doubleSpinBox_03.value())  # 高斯模糊
        min_val = int(self.doubleSpinBox_04.value())  # 最小半径
        max_val = int(self.doubleSpinBox_05.value())  # 最大半径
        # 缓存彩图
        self.bgr_cv_obj = copy.deepcopy(self.picture_cv_obj)
        # 缓存灰度图:
        self.gray_cv_obj = copy.deepcopy(self.main_img_cv_obj)
        # 单通道灰度图:用于变换操作
        gray_img = cv2.cvtColor(self.gray_cv_obj, cv2.COLOR_BGR2GRAY)
        # 模糊图
        blur_img = cv2.GaussianBlur(gray_img, (blur_val, blur_val), 3)

        try:
            # 霍夫圆变换
            circles = cv2.HoughCircles(blur_img, cv2.HOUGH_GRADIENT, 1, min_dist, minRadius=min_val, maxRadius=max_val)
            circles = circles[0]

            num = len(circles) if len(circles) < show_mode else show_mode
            for i in range(num):
                x, y, r = circles[i]
                print('x, y, r',x, y, r)
                # 向灰度图画图
                self.gray_cv_obj = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2BGR)
                cv2.circle(self.gray_cv_obj, (x, y), r, (127, 127, 127), 2)  # 画出外圆
                cv2.circle(self.gray_cv_obj, (x, y), 2, (127, 127, 127), 3)  # 画出圆心

                # 向彩图画图
                cv2.circle(self.bgr_cv_obj, (x, y), r, (0, 255, 0), 2)  # 画出外圆
                cv2.circle(self.bgr_cv_obj, (x, y), 2, (0, 0, 255), 3)  # 画出圆心
        except Exception as  e:
            print('错误：' + str(e))
            return

        self.show_img()


if __name__ == '__main__':
    pass
