# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main_Form.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainForm(object):
    def setupUi(self, MainForm):
        MainForm.setObjectName("MainForm")
        MainForm.resize(1920, 1035)
        self.BrowseButton = QtWidgets.QPushButton(MainForm)
        self.BrowseButton.setGeometry(QtCore.QRect(920, 650, 113, 32))
        self.BrowseButton.setObjectName("BrowseButton")
        self.InputImgae_Viewlabel = QtWidgets.QLabel(MainForm)
        self.InputImgae_Viewlabel.setGeometry(QtCore.QRect(30, 50, 1008, 567))
        self.InputImgae_Viewlabel.setAutoFillBackground(True)
        self.InputImgae_Viewlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.InputImgae_Viewlabel.setObjectName("InputImgae_Viewlabel")
        self.LPL_Viewlabel = QtWidgets.QLabel(MainForm)
        self.LPL_Viewlabel.setGeometry(QtCore.QRect(1130, 50, 721, 251))
        self.LPL_Viewlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.LPL_Viewlabel.setObjectName("LPL_Viewlabel")
        self.IE_OCR_Viewlabel = QtWidgets.QLabel(MainForm)
        self.IE_OCR_Viewlabel.setGeometry(QtCore.QRect(1130, 340, 721, 131))
        self.IE_OCR_Viewlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.IE_OCR_Viewlabel.setObjectName("IE_OCR_Viewlabel")
        self.LPR_View_Province_label = QtWidgets.QLabel(MainForm)
        self.LPR_View_Province_label.setGeometry(QtCore.QRect(1130, 510, 80, 100))
        self.LPR_View_Province_label.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Province_label.setObjectName("LPR_View_Province_label")
        self.LPR_View_Letter_label = QtWidgets.QLabel(MainForm)
        self.LPR_View_Letter_label.setGeometry(QtCore.QRect(1230, 510, 80, 100))
        self.LPR_View_Letter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Letter_label.setObjectName("LPR_View_Letter_label")
        self.LPR_View_Digit_label_1 = QtWidgets.QLabel(MainForm)
        self.LPR_View_Digit_label_1.setGeometry(QtCore.QRect(1320, 510, 80, 100))
        self.LPR_View_Digit_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Digit_label_1.setObjectName("LPR_View_Digit_label_1")
        self.LPR_View_Digit_label_2 = QtWidgets.QLabel(MainForm)
        self.LPR_View_Digit_label_2.setGeometry(QtCore.QRect(1410, 510, 80, 100))
        self.LPR_View_Digit_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Digit_label_2.setObjectName("LPR_View_Digit_label_2")
        self.LPR_View_Digit_label_3 = QtWidgets.QLabel(MainForm)
        self.LPR_View_Digit_label_3.setGeometry(QtCore.QRect(1500, 510, 80, 100))
        self.LPR_View_Digit_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Digit_label_3.setObjectName("LPR_View_Digit_label_3")
        self.LPR_View_Digit_label_4 = QtWidgets.QLabel(MainForm)
        self.LPR_View_Digit_label_4.setGeometry(QtCore.QRect(1590, 510, 80, 100))
        self.LPR_View_Digit_label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Digit_label_4.setObjectName("LPR_View_Digit_label_4")
        self.LPR_View_Digit_label_5 = QtWidgets.QLabel(MainForm)
        self.LPR_View_Digit_label_5.setGeometry(QtCore.QRect(1680, 510, 80, 100))
        self.LPR_View_Digit_label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Digit_label_5.setObjectName("LPR_View_Digit_label_5")
        self.LPR_View_Digit_label_6 = QtWidgets.QLabel(MainForm)
        self.LPR_View_Digit_label_6.setGeometry(QtCore.QRect(1770, 510, 80, 100))
        self.LPR_View_Digit_label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.LPR_View_Digit_label_6.setObjectName("LPR_View_Digit_label_6")

        self.retranslateUi(MainForm)
        self.BrowseButton.clicked.connect(MainForm.showButtonMethod)
        QtCore.QMetaObject.connectSlotsByName(MainForm)

    def retranslateUi(self, MainForm):
        _translate = QtCore.QCoreApplication.translate
        MainForm.setWindowTitle(_translate("MainForm", "16010140020 毕业设计 "))
        self.BrowseButton.setText(_translate("MainForm", "Browse"))
        self.InputImgae_Viewlabel.setText(_translate("MainForm", " Show Input Image Here"))
        self.LPL_Viewlabel.setText(_translate("MainForm", "License_Plate_Localization View"))
        self.IE_OCR_Viewlabel.setText(_translate("MainForm", "IE_OCR_View"))
        self.LPR_View_Province_label.setText(_translate("MainForm", "LPR_Province"))
        self.LPR_View_Letter_label.setText(_translate("MainForm", "LPR_Letter"))
        self.LPR_View_Digit_label_1.setText(_translate("MainForm", "LPR_Digit1"))
        self.LPR_View_Digit_label_2.setText(_translate("MainForm", "LPR_Digit2"))
        self.LPR_View_Digit_label_3.setText(_translate("MainForm", "LPR_Digit3"))
        self.LPR_View_Digit_label_4.setText(_translate("MainForm", "LPR_Digit4"))
        self.LPR_View_Digit_label_5.setText(_translate("MainForm", "LPR_Digit5"))
        self.LPR_View_Digit_label_6.setText(_translate("MainForm", "LPR_Digit"))
