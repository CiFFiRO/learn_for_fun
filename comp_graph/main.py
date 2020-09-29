import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import time
import math


class Basic(QMainWindow):
    def __init__(self, *args):
        super(Basic, self).__init__(*args)
        loadUi('comp_graph/minimal.ui', self)
        self.__counter = 1
        self.__timer_value = 50

    def setupUI(self):
        self.openGLWidget.initializeGL()
        self.openGLWidget.resizeGL(651,551)
        self.openGLWidget.paintGL = self.draw
        timer = QTimer(self)
        timer.timeout.connect(self.openGLWidget.update)
        timer.start(self.__timer_value)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)

        def triangle():
            glBegin(GL_TRIANGLES)
            glVertex3f(-0.5, -0.5, 0)
            glVertex3f(0.5, -0.5, 0)
            glVertex3f(0.0, 0.5, 0)
            glEnd()

        r = 2
        for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            glLoadIdentity()
            glTranslatef(a * 2.0 + a * r * math.cos(math.pi - math.pi / 180 * self.__counter),
                         b * r * math.sin(math.pi - math.pi / 180 * self.__counter), 0)
            glRotatef(a * b * 5 * self.__counter, 0, 0, 1)
            glScalef(1 + 0.005 * self.__counter, 1 + 0.005 * self.__counter, 1)
            glColor3f(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            triangle()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # glOrtho(-10., 10., -10., 10., -1., 1.)
        glFrustum(-2., 2., -5., 5., 1., 3.)
        glTranslatef(0, 0, -2)

        self.__counter += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Basic()
    window.setupUI()
    window.show()
    sys.exit(app.exec_())
