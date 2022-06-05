from PyQt5.QtWidgets import QApplication, QMainWindow
from myWindow import myWindow
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = myWindow()
    window.show()
    sys.exit(app.exec_())
