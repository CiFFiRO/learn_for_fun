import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import time
import math


TIMER_VALUE = 50


def triangle_show(func):
    start = time.time()
    def wrapper():
        func(start)
    return wrapper


@triangle_show
def triangle_show_draw(start):
    counter = (time.time() - start)*1000//TIMER_VALUE
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
        glTranslatef(a*2.0+a*r*math.cos(math.pi-math.pi/180*counter), b*r*math.sin(math.pi-math.pi/180*counter), 0)
        glRotatef(a*b*5 * counter, 0, 0, 1)
        glScalef(1+0.005*counter, 1+0.005*counter, 1)
        glColor3f(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        triangle()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # glOrtho(-10., 10., -10., 10., -1., 1.)
    glFrustum(-2., 2., -5., 5., 1., 3.)
    glTranslatef(0, 0, -2)


class MainWindow(QMainWindow):
    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)
        loadUi('comp_graph/minimal.ui', self)
        self.counter = 1

    def setupUI(self):
        self.openGLWidget.initializeGL()
        self.openGLWidget.resizeGL(651,551)
        self.openGLWidget.paintGL = self.paintGL
        timer = QTimer(self)
        timer.timeout.connect(self.openGLWidget.update)
        timer.start(TIMER_VALUE)

    def paintGL(self):
        triangle_show_draw()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setupUI()
    window.show()
    sys.exit(app.exec_())
