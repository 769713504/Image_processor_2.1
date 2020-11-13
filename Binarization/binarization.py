import cv2
import copy
from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from Binarization.GetAreaWindow import Ui_Form


class AreaBinarization(Ui_Form, QDialog):
    def __init__(self, mainwindow, picture, image):
        super(AreaBinarization, self).__init__()
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
        # 重写ico图标
        icon = QtGui.QIcon()
        ico_path = "./Binarization/logo.ico"
        icon.addPixmap(QtGui.QPixmap(ico_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        # 点击事件
        self.pushButton_1.clicked.connect(self.reload)  # 重载
        self.pushButton_2.clicked.connect(self.return_img)  # 返回
        self.pushButton_3.clicked.connect(self.show_img)  # 原图
        self.pushButton_4.clicked.connect(self.operation)  # 开始变换
        # 双精度数字调节框
        pass
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


    # todo 功能:运行 ########################################################################################################
    def operation(self):
        pass


if __name__ == '__main__':
    pass
