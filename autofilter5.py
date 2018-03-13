# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'autofilter.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

#commit@pratham: All Dialogs functional, video gets displayed after selecting file from filedialog, image still not functional. Exit not working
#finalCommit

from PyQt5 import QtCore, QtWidgets, QtGui
import os
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
import image_sensor.py
try:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    import cv2
except ImportError:
    print("Please install the required packages.")
    sys.exit()

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)



class videoDialog(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.openFileNameDialog()
       
        
 
        self.show()
 
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Video files (*.mp4 *.3gp *.webm *.flv *.avi *.wmv *.mov);;All Files (*)", options=options)
        if fileName:
            print(fileName)
#Here the 'filename' will be provided to the model to process


#The model will return a processes file 'filename' which will get displayed
            cap  =cv2.VideoCapture(fileName)#'/home/pratham/demo2.mp4')
            while(cap.isOpened()):
                ret, frame = cap.read()
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.namedWindow('frame',800*400)
                cv2.imshow('frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
 


class imageDialog(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.openFileNameDialog()
       
        
 
        self.show()
 
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Image files (*.jpg *.jpeg *.png *.tiff *.bmp);;All Files (*)", options=options)
        if fileName:
            print(fileName)
            viewer=ImageViewer(fileName)
#Here the 'filename' will be provided to the model to process


       
           
        
class ImageViewer(QWidget):

    def __init__(self,filename):
        QWidget.__init__(self)
        self.filename = filename #= "/home/pratham/Pictures/creative.jpg"
        print("hello babay"+filename)
        self.setup_ui()

    def setup_ui(self):
        img = cv2.imread(self.filename)
        self.image_label = QLabel()
        if img is None:
            self.image_label.setText("Cannot load the input image.")
        else:
           # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#The model will return a processes file 'filename' which will get displayed     
            cv2.imshow('image',img)
        #     img_ = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        #     self.image_label.setPixmap(QPixmap.fromImage(img_))
        # self.quit_button = QPushButton("Quit")
        # self.quit_button.clicked.connect(self.close)
        # self.main_layout = QVBoxLayout()
        # self.main_layout.addWidget(self.image_label)
        # self.main_layout.addWidget(self.quit_button)
        # self.setLayout(self.main_layout)
        # self.setWindowTitle("OpenCV - Qt Integration")





class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(639, 450)
        Dialog.setMaximumSize(QtCore.QSize(700, 450))
        Dialog.setAutoFillBackground(False)
        Dialog.setStyleSheet(_fromUtf8("background-image:url(./image1.jpg);\n"
" \n"
""))
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(100, 190, 201, 151))
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet(_fromUtf8("font: 24pt \"Times New Roman\";\n"
"color: rgb(255, 255, 255);"))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(360, 190, 191, 151))
        self.pushButton_2.setAutoFillBackground(False)
        self.pushButton_2.setStyleSheet(_fromUtf8("font: 24pt \"Times New Roman\";\n"
"color: rgb(255, 255, 255);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 120, 621, 31))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Times New Roman"))
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setStyleSheet(_fromUtf8("color: rgb(255, 255, 255);"))
        self.label.setObjectName(_fromUtf8("label"))
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(520, 400, 99, 27))
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setStyleSheet(_fromUtf8("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 255), stop:0.166 rgba(255, 255, 0, 255), stop:0.333 rgba(0, 255, 0, 255), stop:0.5 rgba(0, 255, 255, 255), stop:0.666 rgba(0, 0, 255, 255), stop:0.833 rgba(255, 0, 255, 255), stop:1 rgba(255, 0, 0, 255));\n"
"color: rgb(255, 255, 255);\n"
""))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(330, 0, 311, 61))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial Black"))
        font.setPointSize(36)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(_fromUtf8("color: rgb(0, 0, 0);\n"
"color: rgb(255, 0, 0);"))
        self.label_2.setObjectName(_fromUtf8("label_2"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.pushButton.setText(_translate("Dialog", "Image", None))
        self.pushButton.clicked.connect(imageDialog)
   
        self.pushButton_2.setText(_translate("Dialog", "Video", None))
        self.pushButton_2.clicked.connect(videoDialog)

        self.label.setText(_translate("Dialog", "Select media type..", None))
        self.pushButton_3.setText(_translate("Dialog", "Exit", None))
        #self.pushButton_3.clicked.connect(self.close)
        self.label_2.setText(_translate("Dialog", "#AutoFilter", None))

if __name__ == "__main__":
   import sys
   app = QtWidgets.QApplication(sys.argv)
   Dialog = QtWidgets.QDialog()
   ui = Ui_Dialog()
   ui.setupUi(Dialog)
   Dialog.show()
   sys.exit(app.exec_())