
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image,ImageQt,ImageFilter
import numpy as np
from scipy import ndimage, misc
from collections import deque
import cv2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1015, 758)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMaximumSize(QtCore.QSize(1904, 1071))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(1280, 720))
        self.label.setText("")
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select an image to use",QtCore.QDir.homePath(), "Images (*.png *.xpm *.jpg)")
        
        print(str(self.filename))
                
        self.filedir = QtCore.QFileInfo(self.filename).absolutePath()
        self.image_file=cv2.imread(str(self.filename))
        #self.image_file=QtGui.QPixmap(self.filename)
        #self.iqt=ImageQt.ImageQt(self.image_file)
        self.iqt=QtGui.QImage(self.image_file.data, self.image_file.shape[1], self.image_file.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))
        
        #self.img_arr=np.array(self.image_file)
        #print(self.img_arr.shape)
        #print(self.img_arr[50:52,40:50])
        
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.fselectButton = QtWidgets.QPushButton(self.centralwidget)
        self.fselectButton.setObjectName("fselectButton")
        self.verticalLayout.addWidget(self.fselectButton, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.flipButton = QtWidgets.QPushButton(self.centralwidget)
        self.flipButton.setObjectName("flipButton")
        self.verticalLayout.addWidget(self.flipButton, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        #self.datashowButton = QtWidgets.QPushButton(self.centralwidget)
        #self.datashowButton.setObjectName("datashowButton")
        #self.verticalLayout.addWidget(self.datashowButton, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        
        self.brightnessSlider = QtWidgets.QSlider(self.centralwidget)
        self.brightnessSlider.setObjectName(u"brightnessSlider")
        #self.brightnessSlider.setGeometry(QRect(300, 270, 160, 19))
        self.brightnessSlider.setMinimum(-255)
        self.brightnessSlider.setMaximum(255)
        self.brightnessSlider.setOrientation(QtCore.Qt.Horizontal)
        self.verticalLayout.addWidget(self.brightnessSlider, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)

        self.bchangeposButton = QtWidgets.QPushButton(self.centralwidget)
        self.bchangeposButton.setObjectName("bchangeposButton")
        self.verticalLayout.addWidget(self.bchangeposButton, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.bchangenegButton = QtWidgets.QPushButton(self.centralwidget)
        self.bchangenegButton.setObjectName("bchangenegButton")
        self.verticalLayout.addWidget(self.bchangenegButton, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.status = QtWidgets.QLabel(self.centralwidget)
        self.status.setObjectName("status")
        self.verticalLayout.addWidget(self.status)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.flag=1
        self.status.setText(str("Contrast Slider Value: {}".format(self.brightnessSlider.value())))
        self.fselectButton.clicked.connect(self.set_new_image)
        self.flipButton.clicked.connect(self.flip_image)
        #self.bchangeposButton.clicked.connect(self.bchangepos)
        self.bchangeposButton.clicked.connect(self.implemented_canny)
        #self.bchangenegButton.clicked.connect(self.bchangeneg)
        self.bchangenegButton.clicked.connect(self.opencv_canny)
        self.brightnessSlider.sliderMoved.connect(self.cchangeslide)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.fselectButton.setText(_translate("MainWindow", "Select File"))
        self.flipButton.setText(_translate("MainWindow", "Flip"))
        #self.datashowButton.setText(_translate("MainWindow", "Show Data"))
        #self.bchangeposButton.setText(_translate("MainWindow", "Change Brightness +5"))
        self.bchangeposButton.setText(_translate("MainWindow", "Implemented Canny"))
        #self.bchangenegButton.setText(_translate("MainWindow", "Change Brightness -5"))
        self.bchangenegButton.setText(_translate("MainWindow", "OpenCV Canny"))
        self.status.setText(_translate("MainWindow", "Value:"))

    def prep_img_conv(self,img_target,height_req,width_req):
        prep_img_height=height_req[0]*height_req[1]
        prep_img_width=width_req[0]*width_req[1]
        prep_arr=np.zeros((prep_img_height,prep_img_width))
        print(prep_img_height)
        print(prep_img_width)
        print(width_req)
        row=0
        for i in range(0,height_req[0]):
            for j in range(0,height_req[1]):
                prep_arr[row,:]=np.ravel(img_target[i:i+width_req[0],j:j+width_req[1]])
                row+=1
                #print(window.shape)
        
        return prep_arr
    
    def img_convolve(self,image_arr,kernel):
        img_height=image_arr.shape[0]
        img_width=image_arr.shape[1]
        pad_size=kernel.shape[0]//2
        temp_image_arr=np.pad(image_arr,pad_size,mode="edge")
        print(image_arr.shape)
        print(temp_image_arr.shape)
        prep_image_arr=self.prep_img_conv(temp_image_arr,image_arr.shape,kernel.shape)
        kernel_flat=np.ravel(kernel)
        print(prep_image_arr.shape)
        print(kernel_flat.shape)
        result_img_arr=image_arr
        return result_img_arr

    
    def set_new_image(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select an image to use",self.filedir, "Images (*.png *.xpm *.jpg)")
        self.filedir = QtCore.QFileInfo(self.filename).absolutePath()
        self.image_file=cv2.imread(str(self.filename))
        self.iqt=QtGui.QImage(self.image_file.data, self.image_file.shape[1], self.image_file.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))
        self.brightnessSlider.setValue(0)
        self.status.setText(str("Slider Value: {}".format(self.brightnessSlider.value())))
       
    def flip_image(self):
        
        if self.flag % 2 == 0 :
            self.label.setMaximumSize(QtCore.QSize(1280, 720))
        else:
            self.label.setMaximumSize(QtCore.QSize(720, 1280))
        self.flag+=1

    def bchangepos(self):

        self.image_file=self.image_file.point(lambda i:i+5)
        self.iqt=ImageQt.ImageQt(self.image_file)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))

    def bchangeneg(self):

        self.image_file=self.image_file.point(lambda i:i-5)
        self.iqt=ImageQt.ImageQt(self.image_file)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))


    def cchangeslide(self, val):

        val/=5
        cfactor= float(259*(255+val)/(255*(259-val)))
        self.image_file_2=cv2.addWeighted(self.image_file,cfactor,self.image_file,0,float(128-(cfactor*128)))
        self.iqt=QtGui.QImage(self.image_file_2.data, self.image_file_2.shape[1], self.image_file_2.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))
        self.status.setText(str("Contrast Slider Value: {}".format(self.brightnessSlider.value())))
    
    def cull(self,arr):

        if(arr<0):
            return 0
        elif(arr>255):
            return 255
        else:
            return arr
    
    def hyster(self,img,low,high):

        height=img.shape[0]
        width=img.shape[1]
        bst_stack=deque()
        strong=255
        weak=150
        for i in range(0,height):
            for j in range(0,width):
                val=img[i,j]
                if(val<low):
                    img[i,j]=0
                elif(val>=high):
                    img[i,j]=strong
                    bst_stack.append((i,j))
                else:
                    img[i,j]=weak
        
        img_ret=np.zeros((height,width))
        
        while bst_stack:
            
            idx_i,idx_j=bst_stack.pop()
            img_ret[idx_i,idx_j]=strong
            neighbors=[(idx_i-1,idx_j-1),(idx_i-1,idx_j),(idx_i-1,idx_j+1),(idx_i,idx_j-1),(idx_i,idx_j+1),(idx_i+1,idx_j-1),(idx_i+1,idx_j),(idx_i+1,idx_j+1)]
            for a,b in neighbors:
                if(img[a,b]==weak):
                    img[a,b]=strong
                    img_ret[a,b]=strong
                    bst_stack.append((a,b))
        
        return img_ret

    
    def edge_suppress(self,img_mag,img_theta):

        height=img_mag.shape[0]
        width=img_mag.shape[1]
        img_edge_sup=np.zeros((height,width))
        PI=180
        for i in range(1,height-1):
            for j in range(1,width-1):
                angle=img_theta[i,j]
                cur_val=img_mag[i,j]
                if( (0<=angle<PI/8) or (7*PI/8<=angle<9*PI/8) or (15*PI/8<=angle<=2*PI) ):
                    comp_neg=img_mag[i-1,j]
                    comp_pos=img_mag[i+1,j]
                elif( (PI/8<=angle<3*PI/8) or (9*PI/8<=angle<11*PI/8) ):
                    comp_neg=img_mag[i-1,j-1]
                    comp_pos=img_mag[i+1,j+1]
                elif( (3*PI/8<=angle<5*PI/8) or (11*PI/8<=angle<13*PI/8) ):
                    comp_neg=img_mag[i,j-1]
                    comp_pos=img_mag[i,j+1]
                else:
                    comp_neg=img_mag[i+1,j-1]
                    comp_pos=img_mag[i-1,j+1]
                
                if(cur_val>=comp_neg and cur_val>=comp_pos):
                    img_edge_sup[i,j]=cur_val
        
        return img_edge_sup
    
    def implemented_canny(self):
        
        self.img_grayscale=cv2.cvtColor(self.image_file,cv2.COLOR_BGR2GRAY)
        self.img_arr=cv2.GaussianBlur(self.img_grayscale,(0,0),2)
        print(self.img_arr.shape)
        self.img_arr_conv_vedge=cv2.Sobel(self.img_arr,cv2.CV_64F,1,0)
        self.img_arr_conv_hedge=cv2.Sobel(self.img_arr,cv2.CV_64F,0,1)
        self.img_sbl_mag=np.sqrt((np.square(self.img_arr_conv_vedge)) + (np.square(self.img_arr_conv_hedge)))
        self.img_sbl_theta=np.arctan2(self.img_arr_conv_vedge,self.img_arr_conv_hedge)
        PI=180
        self.img_sbl_theta=np.rad2deg(self.img_sbl_theta)
        self.img_sbl_theta+=PI
        self.img_arr_fin=self.edge_suppress(self.img_sbl_mag,self.img_sbl_theta)
        low=15
        high=50
        self.img_arr_fin=self.hyster(self.img_arr_fin,low,high)
        self.image_file_2=cv2.convertScaleAbs(self.img_arr_fin)
        self.iqt=QtGui.QImage(self.image_file_2.data, self.image_file_2.shape[1], self.image_file_2.shape[0], QtGui.QImage.Format_Grayscale8)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))
    
    def opencv_canny(self):
        self.img_grayscale=cv2.cvtColor(self.image_file,cv2.COLOR_BGR2GRAY)
        self.img_arr=cv2.GaussianBlur(self.img_grayscale,(0,0),2)
        self.image_file2=cv2.Canny(self.img_arr,15,50)
        self.iqt=QtGui.QImage(self.image_file2.data, self.image_file2.shape[1], self.image_file2.shape[0], QtGui.QImage.Format_Grayscale8)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.iqt))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
