from GUI import Main_Form
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class myMainForm(QMainWindow):
    def __init__(self):
        super(myMainForm, self).__init__()
        self.Main_Form_Ui = Main_Form.Ui_MainForm()
        self.Main_Form_Ui.setupUi(self)

    def showButtonMethod(self):
        file_name = QFileDialog.getOpenFileName(self, "Open File", "./", "jpg (*.jpg)")
        image_path = file_name[0]
        if (file_name[0] == ""):
            QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
        pix = QPixmap(image_path)
        self.Main_Form_Ui.InputImgae_Viewlabel.setPixmap(pix)

