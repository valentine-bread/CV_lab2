from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 as cv
import numpy as np

def conv_cv_to_qpixmap(img_cv):
    height, width, channel = img_cv.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(img_cv.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap(qImg)

def process_image_MT(in_img, template):
    # in_img = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(in_img,template,cv.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv.minMaxLoc(res)
    return max_loc, (max_loc[0] + w, max_loc[1] + h) 

def process_image_SIFT(img1, img2):
    MIN_MATCH_COUNT = 10
    MAX_DISTANCE = 0.6
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)                # Здесь kp будет списком ключевых точек, а des — масивом формата numpy.(Количество ключевых точек) × 128
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # Выбор алгоритма для sift
    search_params = dict(checks = 50)                           # Количество обходов дерева 
    flann = cv.FlannBasedMatcher(index_params, search_params)   # (FLANN) Быстрая библиотека для приближенных ближайших соседей
    matches = flann.knnMatch(des1,des2,k=2)     

    good = []
    for m,n in matches:
        if m.distance < MAX_DISTANCE*n.distance:         # distance - Расстояние между дескрипторами
            good.append(m)     
            
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)  # queryIdx - Index of the descriptor in query descriptors
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)  # trainIdx — Index of the descriptor in train descriptors
        _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)    
         
        pts = src_pts[mask==1]
        min_x, min_y = np.int32(pts.min(axis=0))
        max_x, max_y = np.int32(pts.max(axis=0))  
        return (min_x, min_y), (max_x,max_y)

    else:
        raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(566, 433)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.image = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image.sizePolicy().hasHeightForWidth())
        self.image.setSizePolicy(sizePolicy)
        self.image.setText("")
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        self.horizontalLayout.addWidget(self.image)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.template_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.template_2.sizePolicy().hasHeightForWidth())
        self.template_2.setSizePolicy(sizePolicy)
        self.template_2.setText("")
        self.template_2.setScaledContents(True)
        self.template_2.setObjectName("template_2")
        self.verticalLayout_2.addWidget(self.template_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
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


        
        self.comboBox.addItem('Template Matching')
        self.comboBox.addItem('Feature Matching SIFT')

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.spinBox.valueChanged.connect(self.process)
        self.comboBox.currentIndexChanged.connect(self.process)
        self.spinBox.setValue(1)
        self.process()
      
    def process(self):
        self.statusbar.showMessage("") 
        i = self.spinBox.value()
        in_img = cv.imread('Lab2/img/' + str(i) +'.jpg',0)
        template = cv.imread('Lab2/template/' + str(i) +'.jpg',0)
        if type(in_img) != np.ndarray and type(template) != np.ndarray:
            self.statusbar.showMessage("Image not found") 
            return
        try:
            if self.comboBox.currentText() == 'Template Matching':
                foo = process_image_MT
                # in_img = process_image_MT(in_img, template)
            elif self.comboBox.currentText() == 'Feature Matching SIFT':    
                foo = process_image_SIFT       
                # in_img = process_image_SIFT(in_img, template)
            x,y = foo(in_img, template)
            
            in_img = cv.imread('Lab2/img/' + str(i) +'.jpg')
            temp = cv.imread('Lab2/template/' + str(i) +'.jpg')
            cv.rectangle(in_img, x, y, (0,0,255), 2)
            self.image.setPixmap(conv_cv_to_qpixmap(in_img))
            self.template_2.setPixmap(conv_cv_to_qpixmap(temp))
        except Exception as e:
            self.statusbar.showMessage(str(e))
            print(str(e))

    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


