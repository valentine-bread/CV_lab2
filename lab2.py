from asyncio.windows_events import NULL
from pickle import TRUE
import string
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def conv_cv_to_qpixmap(img_cv):
    height, width, channel = img_cv.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(img_cv.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap(qImg)

def process_image_MT(in_img, template):
    img_gray = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    cv.rectangle(in_img, top_left, (top_left[0] + w, top_left[1] + h), 255, 2)
    return in_img   

def process_image_SIFT(img1, img2):
    MIN_MATCH_COUNT = 10
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)               
        pts = src_pts[mask==1]
        min_x, min_y = np.int32(pts.min(axis=0))
        max_x, max_y = np.int32(pts.max(axis=0))  
        # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)     
        cv.rectangle(img1,(min_x, min_y), (max_x,max_y), (0,0,255), 2)
        return img1 
        
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None    
        return NULL
        

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(566, 433)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.image = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image.sizePolicy().hasHeightForWidth())
        self.image.setSizePolicy(sizePolicy)
        self.image.setText("")
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        self.verticalLayout.addWidget(self.image)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 0, 1, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.spinBox.setValue(1)
        self.comboBox.addItem('Template Matching')
        self.comboBox.addItem('Feature Matching SIFT')

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.spinBox.valueChanged.connect(self.process)
        self.comboBox.currentIndexChanged.connect(self.process)
        self.process()
      
    def process(self):
        self.statusbar.showMessage("") 
        i = self.spinBox.value()
        in_img = cv.imread('Lab2/img/' + str(i) +'.jpg')
        template = cv.imread('Lab2/template/' + str(i) +'.jpg',0)
        if type(in_img) != np.ndarray and type(template) != np.ndarray:
            self.statusbar.showMessage("NoN") 
            return
        if self.comboBox.currentText() == 'Template Matching':
            in_img = process_image_MT(in_img, template)
        elif self.comboBox.currentText() == 'Feature Matching SIFT':
            
            in_img = process_image_SIFT(in_img, template)
        self.image.setPixmap(conv_cv_to_qpixmap(in_img))

    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


