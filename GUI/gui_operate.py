import sys
from PyQt5.QtWidgets import QApplication
from GUI import my_MainForm_class

def Gui_Generator():
    """进行车牌识别Gui_Generator"""

    app = QApplication(sys.argv)
    MainWindow = my_MainForm_class.myMainForm()
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    Gui_Generator()