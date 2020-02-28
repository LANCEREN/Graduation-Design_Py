import sys
from PyQt5.QtWidgets import QApplication
from GUI import MyMainFormClass

def gui_generator():
    """进行车牌识别gui_generator"""

    app = QApplication(sys.argv)
    MainWindow = MyMainFormClass.myMainForm()
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    gui_generator()