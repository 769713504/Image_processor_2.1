# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1231, 752)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("main.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_14.setGeometry(QtCore.QRect(661, 83, 110, 35))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_27 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_27.setGeometry(QtCore.QRect(900, 48, 110, 35))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.pushButton_27.setFont(font)
        self.pushButton_27.setObjectName("pushButton_27")
        self.pushButton_00 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_00.setGeometry(QtCore.QRect(10, 526, 160, 120))
        self.pushButton_00.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_00.setText("")
        self.pushButton_00.setObjectName("pushButton_00")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 6, 640, 480))
        self.label.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(1020, 156, 111, 30))
        self.comboBox_4.setStyleSheet("")
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(779, 9, 110, 35))
        self.pushButton_11.setObjectName("pushButton_11")
        self.label_01 = QtWidgets.QLabel(self.centralwidget)
        self.label_01.setGeometry(QtCore.QRect(170, 526, 160, 120))
        self.label_01.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_01.setText("")
        self.label_01.setObjectName("label_01")
        self.label_03 = QtWidgets.QLabel(self.centralwidget)
        self.label_03.setGeometry(QtCore.QRect(490, 526, 160, 120))
        self.label_03.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_03.setText("")
        self.label_03.setObjectName("label_03")
        self.pushButton_23 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_23.setGeometry(QtCore.QRect(661, 416, 110, 35))
        self.pushButton_23.setObjectName("pushButton_23")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 486, 641, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton.setChecked(True)
        self.radioButton.setAutoRepeat(False)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout.addWidget(self.radioButton)
        self.radioButton_00 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_00.setObjectName("radioButton_00")
        self.horizontalLayout.addWidget(self.radioButton_00)
        self.radioButton_01 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_01.setObjectName("radioButton_01")
        self.horizontalLayout.addWidget(self.radioButton_01)
        self.radioButton_02 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_02.setObjectName("radioButton_02")
        self.horizontalLayout.addWidget(self.radioButton_02)
        self.radioButton_03 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_03.setObjectName("radioButton_03")
        self.horizontalLayout.addWidget(self.radioButton_03)
        self.radioButton_04 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_04.setObjectName("radioButton_04")
        self.horizontalLayout.addWidget(self.radioButton_04)
        self.radioButton_05 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_05.setObjectName("radioButton_05")
        self.horizontalLayout.addWidget(self.radioButton_05)
        self.radioButton_06 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_06.setObjectName("radioButton_06")
        self.horizontalLayout.addWidget(self.radioButton_06)
        self.radioButton_07 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_07.setObjectName("radioButton_07")
        self.horizontalLayout.addWidget(self.radioButton_07)
        self.radioButton_08 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_08.setObjectName("radioButton_08")
        self.horizontalLayout.addWidget(self.radioButton_08)
        self.radioButton_09 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_09.setObjectName("radioButton_09")
        self.horizontalLayout.addWidget(self.radioButton_09)
        self.radioButton_10 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_10.setObjectName("radioButton_10")
        self.horizontalLayout.addWidget(self.radioButton_10)
        self.pushButton_22 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_22.setGeometry(QtCore.QRect(661, 379, 110, 35))
        self.pushButton_22.setObjectName("pushButton_22")
        self.doubleSpinBox_00 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_00.setGeometry(QtCore.QRect(779, 159, 110, 30))
        self.doubleSpinBox_00.setAccelerated(True)
        self.doubleSpinBox_00.setDecimals(0)
        self.doubleSpinBox_00.setMaximum(255.0)
        self.doubleSpinBox_00.setSingleStep(1.0)
        self.doubleSpinBox_00.setProperty("value", 127.0)
        self.doubleSpinBox_00.setObjectName("doubleSpinBox_00")
        self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_19.setGeometry(QtCore.QRect(661, 268, 110, 35))
        self.pushButton_19.setObjectName("pushButton_19")
        self.label_09 = QtWidgets.QLabel(self.centralwidget)
        self.label_09.setGeometry(QtCore.QRect(810, 586, 80, 60))
        self.label_09.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_09.setText("")
        self.label_09.setObjectName("label_09")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(1020, 16, 110, 35))
        self.checkBox.setAcceptDrops(False)
        self.checkBox.setStyleSheet("")
        self.checkBox.setChecked(False)
        self.checkBox.setProperty("tabletTracking", False)
        self.checkBox.setObjectName("checkBox")
        self.pushButton_05 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_05.setGeometry(QtCore.QRect(650, 586, 80, 60))
        self.pushButton_05.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_05.setText("")
        self.pushButton_05.setObjectName("pushButton_05")
        self.doubleSpinBox_05 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_05.setGeometry(QtCore.QRect(779, 344, 110, 30))
        self.doubleSpinBox_05.setAccelerated(True)
        self.doubleSpinBox_05.setDecimals(0)
        self.doubleSpinBox_05.setMinimum(1.0)
        self.doubleSpinBox_05.setMaximum(1000.0)
        self.doubleSpinBox_05.setSingleStep(1.0)
        self.doubleSpinBox_05.setProperty("value", 5.0)
        self.doubleSpinBox_05.setObjectName("doubleSpinBox_05")
        self.pushButton_29 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_29.setGeometry(QtCore.QRect(900, 84, 110, 35))
        self.pushButton_29.setObjectName("pushButton_29")
        self.comboBox_1 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_1.setGeometry(QtCore.QRect(779, 86, 110, 30))
        self.comboBox_1.setObjectName("comboBox_1")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.label_06 = QtWidgets.QLabel(self.centralwidget)
        self.label_06.setGeometry(QtCore.QRect(730, 526, 80, 60))
        self.label_06.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_06.setText("")
        self.label_06.setObjectName("label_06")
        self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_12.setGeometry(QtCore.QRect(661, 46, 110, 35))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_33 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_33.setGeometry(QtCore.QRect(900, 190, 230, 70))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_33.setFont(font)
        self.pushButton_33.setObjectName("pushButton_33")
        self.pushButton_07 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_07.setGeometry(QtCore.QRect(730, 586, 80, 60))
        self.pushButton_07.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_07.setText("")
        self.pushButton_07.setObjectName("pushButton_07")
        self.label_02 = QtWidgets.QLabel(self.centralwidget)
        self.label_02.setGeometry(QtCore.QRect(330, 526, 160, 120))
        self.label_02.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_02.setText("")
        self.label_02.setObjectName("label_02")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(779, 123, 110, 30))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.pushButton_30 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_30.setGeometry(QtCore.QRect(1020, 84, 110, 35))
        self.pushButton_30.setObjectName("pushButton_30")
        self.label_05 = QtWidgets.QLabel(self.centralwidget)
        self.label_05.setGeometry(QtCore.QRect(650, 586, 80, 60))
        self.label_05.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_05.setText("")
        self.label_05.setObjectName("label_05")
        self.pushButton_20 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_20.setGeometry(QtCore.QRect(661, 305, 110, 35))
        self.pushButton_20.setObjectName("pushButton_20")
        self.pushButton_25 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_25.setGeometry(QtCore.QRect(661, 490, 110, 35))
        self.pushButton_25.setObjectName("pushButton_25")
        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_13.setGeometry(QtCore.QRect(779, 46, 110, 35))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_04 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_04.setGeometry(QtCore.QRect(650, 526, 80, 60))
        self.pushButton_04.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_04.setText("")
        self.pushButton_04.setObjectName("pushButton_04")
        self.doubleSpinBox_04 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_04.setGeometry(QtCore.QRect(779, 307, 110, 30))
        self.doubleSpinBox_04.setAccelerated(True)
        self.doubleSpinBox_04.setDecimals(0)
        self.doubleSpinBox_04.setMinimum(1.0)
        self.doubleSpinBox_04.setMaximum(100.0)
        self.doubleSpinBox_04.setSingleStep(1.0)
        self.doubleSpinBox_04.setProperty("value", 3.0)
        self.doubleSpinBox_04.setObjectName("doubleSpinBox_04")
        self.doubleSpinBox_02 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_02.setGeometry(QtCore.QRect(779, 233, 110, 30))
        self.doubleSpinBox_02.setAccelerated(True)
        self.doubleSpinBox_02.setDecimals(0)
        self.doubleSpinBox_02.setMinimum(1.0)
        self.doubleSpinBox_02.setMaximum(30.0)
        self.doubleSpinBox_02.setSingleStep(1.0)
        self.doubleSpinBox_02.setProperty("value", 2.0)
        self.doubleSpinBox_02.setObjectName("doubleSpinBox_02")
        self.pushButton_09 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_09.setGeometry(QtCore.QRect(810, 586, 80, 60))
        self.pushButton_09.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_09.setText("")
        self.pushButton_09.setObjectName("pushButton_09")
        self.pushButton_16 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_16.setGeometry(QtCore.QRect(661, 157, 110, 35))
        self.pushButton_16.setObjectName("pushButton_16")
        self.doubleSpinBox_01 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_01.setGeometry(QtCore.QRect(779, 196, 110, 30))
        self.doubleSpinBox_01.setAccelerated(True)
        self.doubleSpinBox_01.setDecimals(0)
        self.doubleSpinBox_01.setMinimum(1.0)
        self.doubleSpinBox_01.setMaximum(30.0)
        self.doubleSpinBox_01.setSingleStep(1.0)
        self.doubleSpinBox_01.setProperty("value", 2.0)
        self.doubleSpinBox_01.setObjectName("doubleSpinBox_01")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(661, 9, 110, 35))
        self.pushButton_10.setObjectName("pushButton_10")
        self.doubleSpinBox_08 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_08.setGeometry(QtCore.QRect(779, 455, 110, 30))
        self.doubleSpinBox_08.setAccelerated(True)
        self.doubleSpinBox_08.setDecimals(0)
        self.doubleSpinBox_08.setMinimum(-360.0)
        self.doubleSpinBox_08.setMaximum(360.0)
        self.doubleSpinBox_08.setSingleStep(1.0)
        self.doubleSpinBox_08.setProperty("value", 0.0)
        self.doubleSpinBox_08.setObjectName("doubleSpinBox_08")
        self.doubleSpinBox_09 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_09.setGeometry(QtCore.QRect(779, 492, 110, 30))
        self.doubleSpinBox_09.setAccelerated(True)
        self.doubleSpinBox_09.setDecimals(1)
        self.doubleSpinBox_09.setMinimum(0.1)
        self.doubleSpinBox_09.setMaximum(10.0)
        self.doubleSpinBox_09.setSingleStep(0.1)
        self.doubleSpinBox_09.setProperty("value", 1.0)
        self.doubleSpinBox_09.setObjectName("doubleSpinBox_09")
        self.pushButton_17 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_17.setGeometry(QtCore.QRect(661, 194, 110, 35))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_18.setGeometry(QtCore.QRect(661, 231, 110, 35))
        self.pushButton_18.setObjectName("pushButton_18")
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(1020, 123, 110, 30))
        self.comboBox_3.setStyleSheet("")
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.pushButton_26 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_26.setGeometry(QtCore.QRect(900, 12, 110, 35))
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_21 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_21.setGeometry(QtCore.QRect(661, 342, 110, 35))
        self.pushButton_21.setObjectName("pushButton_21")
        self.label_07 = QtWidgets.QLabel(self.centralwidget)
        self.label_07.setGeometry(QtCore.QRect(730, 586, 80, 60))
        self.label_07.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_07.setText("")
        self.label_07.setObjectName("label_07")
        self.label_08 = QtWidgets.QLabel(self.centralwidget)
        self.label_08.setGeometry(QtCore.QRect(810, 526, 80, 60))
        self.label_08.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_08.setText("")
        self.label_08.setObjectName("label_08")
        self.pushButton_15 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_15.setGeometry(QtCore.QRect(661, 120, 110, 35))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_08 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_08.setGeometry(QtCore.QRect(810, 526, 80, 60))
        self.pushButton_08.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_08.setText("")
        self.pushButton_08.setObjectName("pushButton_08")
        self.pushButton_34 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_34.setGeometry(QtCore.QRect(900, 260, 230, 70))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_34.setFont(font)
        self.pushButton_34.setObjectName("pushButton_34")
        self.label_00 = QtWidgets.QLabel(self.centralwidget)
        self.label_00.setGeometry(QtCore.QRect(10, 526, 160, 120))
        self.label_00.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_00.setText("")
        self.label_00.setObjectName("label_00")
        self.pushButton_01 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_01.setGeometry(QtCore.QRect(170, 526, 160, 120))
        self.pushButton_01.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_01.setText("")
        self.pushButton_01.setObjectName("pushButton_01")
        self.pushButton_24 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_24.setGeometry(QtCore.QRect(661, 453, 110, 35))
        self.pushButton_24.setObjectName("pushButton_24")
        self.doubleSpinBox_03 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_03.setGeometry(QtCore.QRect(779, 270, 110, 30))
        self.doubleSpinBox_03.setAccelerated(True)
        self.doubleSpinBox_03.setDecimals(0)
        self.doubleSpinBox_03.setMinimum(1.0)
        self.doubleSpinBox_03.setMaximum(100.0)
        self.doubleSpinBox_03.setSingleStep(1.0)
        self.doubleSpinBox_03.setProperty("value", 3.0)
        self.doubleSpinBox_03.setObjectName("doubleSpinBox_03")
        self.pushButton_32 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_32.setGeometry(QtCore.QRect(900, 156, 60, 35))
        self.pushButton_32.setStyleSheet("")
        self.pushButton_32.setObjectName("pushButton_32")
        self.doubleSpinBox_07 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_07.setGeometry(QtCore.QRect(779, 418, 110, 30))
        self.doubleSpinBox_07.setAccelerated(True)
        self.doubleSpinBox_07.setDecimals(0)
        self.doubleSpinBox_07.setMinimum(1.0)
        self.doubleSpinBox_07.setMaximum(1000.0)
        self.doubleSpinBox_07.setSingleStep(1.0)
        self.doubleSpinBox_07.setProperty("value", 5.0)
        self.doubleSpinBox_07.setObjectName("doubleSpinBox_07")
        self.pushButton_28 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_28.setGeometry(QtCore.QRect(1020, 48, 110, 35))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.pushButton_28.setFont(font)
        self.pushButton_28.setObjectName("pushButton_28")
        self.label_04 = QtWidgets.QLabel(self.centralwidget)
        self.label_04.setGeometry(QtCore.QRect(650, 526, 80, 60))
        self.label_04.setStyleSheet("background-color: rgba(250, 255, 255,150);")
        self.label_04.setText("")
        self.label_04.setObjectName("label_04")
        self.doubleSpinBox_06 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_06.setGeometry(QtCore.QRect(779, 381, 110, 30))
        self.doubleSpinBox_06.setAccelerated(True)
        self.doubleSpinBox_06.setDecimals(0)
        self.doubleSpinBox_06.setMinimum(1.0)
        self.doubleSpinBox_06.setMaximum(1000.0)
        self.doubleSpinBox_06.setSingleStep(1.0)
        self.doubleSpinBox_06.setProperty("value", 5.0)
        self.doubleSpinBox_06.setObjectName("doubleSpinBox_06")
        self.pushButton_02 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_02.setGeometry(QtCore.QRect(330, 526, 160, 120))
        self.pushButton_02.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_02.setText("")
        self.pushButton_02.setObjectName("pushButton_02")
        self.pushButton_06 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_06.setGeometry(QtCore.QRect(730, 526, 80, 60))
        self.pushButton_06.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_06.setText("")
        self.pushButton_06.setObjectName("pushButton_06")
        self.doubleSpinBox_10 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_10.setGeometry(QtCore.QRect(960, 156, 50, 35))
        self.doubleSpinBox_10.setStyleSheet("")
        self.doubleSpinBox_10.setAccelerated(True)
        self.doubleSpinBox_10.setDecimals(0)
        self.doubleSpinBox_10.setMinimum(1.0)
        self.doubleSpinBox_10.setMaximum(49.0)
        self.doubleSpinBox_10.setSingleStep(2.0)
        self.doubleSpinBox_10.setProperty("value", 3.0)
        self.doubleSpinBox_10.setObjectName("doubleSpinBox_10")
        self.pushButton_03 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_03.setGeometry(QtCore.QRect(490, 526, 160, 120))
        self.pushButton_03.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"")
        self.pushButton_03.setText("")
        self.pushButton_03.setObjectName("pushButton_03")
        self.pushButton_31 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_31.setGeometry(QtCore.QRect(900, 120, 110, 35))
        self.pushButton_31.setStyleSheet("")
        self.pushButton_31.setObjectName("pushButton_31")
        self.pushButton_35 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_35.setGeometry(QtCore.QRect(900, 330, 230, 70))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton_35.setFont(font)
        self.pushButton_35.setObjectName("pushButton_35")
        self.label_0 = QtWidgets.QLabel(self.centralwidget)
        self.label_0.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.label_0.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_0.setText("")
        self.label_0.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_0.setObjectName("label_0")
        self.label_05.raise_()
        self.label_07.raise_()
        self.label_00.raise_()
        self.label_04.raise_()
        self.pushButton_14.raise_()
        self.pushButton_27.raise_()
        self.pushButton_00.raise_()
        self.label.raise_()
        self.comboBox_4.raise_()
        self.pushButton_11.raise_()
        self.label_01.raise_()
        self.label_03.raise_()
        self.pushButton_23.raise_()
        self.horizontalLayoutWidget.raise_()
        self.pushButton_22.raise_()
        self.doubleSpinBox_00.raise_()
        self.pushButton_19.raise_()
        self.label_09.raise_()
        self.checkBox.raise_()
        self.pushButton_05.raise_()
        self.doubleSpinBox_05.raise_()
        self.pushButton_29.raise_()
        self.comboBox_1.raise_()
        self.label_06.raise_()
        self.pushButton_12.raise_()
        self.pushButton_33.raise_()
        self.pushButton_07.raise_()
        self.label_02.raise_()
        self.comboBox_2.raise_()
        self.pushButton_30.raise_()
        self.pushButton_20.raise_()
        self.pushButton_25.raise_()
        self.pushButton_13.raise_()
        self.pushButton_04.raise_()
        self.doubleSpinBox_04.raise_()
        self.doubleSpinBox_02.raise_()
        self.pushButton_09.raise_()
        self.pushButton_16.raise_()
        self.doubleSpinBox_01.raise_()
        self.pushButton_10.raise_()
        self.doubleSpinBox_08.raise_()
        self.doubleSpinBox_09.raise_()
        self.pushButton_17.raise_()
        self.pushButton_18.raise_()
        self.comboBox_3.raise_()
        self.pushButton_26.raise_()
        self.pushButton_21.raise_()
        self.label_08.raise_()
        self.pushButton_15.raise_()
        self.pushButton_08.raise_()
        self.pushButton_34.raise_()
        self.pushButton_01.raise_()
        self.pushButton_24.raise_()
        self.doubleSpinBox_03.raise_()
        self.pushButton_32.raise_()
        self.doubleSpinBox_07.raise_()
        self.pushButton_28.raise_()
        self.doubleSpinBox_06.raise_()
        self.pushButton_02.raise_()
        self.pushButton_06.raise_()
        self.doubleSpinBox_10.raise_()
        self.pushButton_03.raise_()
        self.pushButton_31.raise_()
        self.pushButton_35.raise_()
        self.label_0.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1231, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.comboBox_4.setCurrentIndex(1)
        self.comboBox_1.setCurrentIndex(1)
        self.comboBox_2.setCurrentIndex(0)
        self.comboBox_3.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像处理"))
        self.pushButton_14.setText(_translate("MainWindow", "图像翻转"))
        self.pushButton_27.setText(_translate("MainWindow", "彩色直方图均衡化"))
        self.comboBox_4.setCurrentText(_translate("MainWindow", "高斯模糊"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "均值模糊"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "高斯模糊"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "中值模糊"))
        self.pushButton_11.setText(_translate("MainWindow", "保存图片"))
        self.pushButton_23.setText(_translate("MainWindow", "黑帽运算"))
        self.radioButton.setText(_translate("MainWindow", "原图"))
        self.radioButton_00.setText(_translate("MainWindow", "Gray"))
        self.radioButton_01.setText(_translate("MainWindow", "B"))
        self.radioButton_02.setText(_translate("MainWindow", "G"))
        self.radioButton_03.setText(_translate("MainWindow", "R"))
        self.radioButton_04.setText(_translate("MainWindow", "H"))
        self.radioButton_05.setText(_translate("MainWindow", "S"))
        self.radioButton_06.setText(_translate("MainWindow", "V"))
        self.radioButton_07.setText(_translate("MainWindow", "L"))
        self.radioButton_08.setText(_translate("MainWindow", "A"))
        self.radioButton_09.setText(_translate("MainWindow", "B"))
        self.radioButton_10.setText(_translate("MainWindow", "All"))
        self.pushButton_22.setText(_translate("MainWindow", "顶帽运算"))
        self.pushButton_19.setText(_translate("MainWindow", "开运算"))
        self.checkBox.setText(_translate("MainWindow", "操作叠加"))
        self.pushButton_29.setText(_translate("MainWindow", "阈值翻转"))
        self.comboBox_1.setCurrentText(_translate("MainWindow", "左右镜像"))
        self.comboBox_1.setItemText(0, _translate("MainWindow", "上下镜像"))
        self.comboBox_1.setItemText(1, _translate("MainWindow", "左右镜像"))
        self.pushButton_12.setText(_translate("MainWindow", "显示原图"))
        self.pushButton_33.setText(_translate("MainWindow", "霍夫圆形变换"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "标准自适应"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "高斯自适应"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "均值自适应"))
        self.pushButton_30.setText(_translate("MainWindow", "显示差异"))
        self.pushButton_20.setText(_translate("MainWindow", "闭运算"))
        self.pushButton_25.setText(_translate("MainWindow", "图像缩放"))
        self.pushButton_13.setText(_translate("MainWindow", "置入当前图"))
        self.pushButton_16.setText(_translate("MainWindow", "二值化"))
        self.pushButton_10.setText(_translate("MainWindow", "打开图片"))
        self.pushButton_17.setText(_translate("MainWindow", "腐蚀"))
        self.pushButton_18.setText(_translate("MainWindow", "膨胀"))
        self.comboBox_3.setCurrentText(_translate("MainWindow", "平滑"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "锐利"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "平滑"))
        self.pushButton_26.setText(_translate("MainWindow", "清空图片"))
        self.pushButton_21.setText(_translate("MainWindow", "形态梯度"))
        self.pushButton_15.setText(_translate("MainWindow", "自适应二值化"))
        self.pushButton_34.setText(_translate("MainWindow", "霍夫直线变换"))
        self.pushButton_24.setText(_translate("MainWindow", "图像旋转"))
        self.pushButton_32.setText(_translate("MainWindow", "模糊"))
        self.pushButton_28.setText(_translate("MainWindow", "灰度直方图均衡化"))
        self.pushButton_31.setText(_translate("MainWindow", "锐化"))
        self.pushButton_35.setText(_translate("MainWindow", "区域二值化"))
