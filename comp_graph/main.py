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
        self.openGLWidget.resizeGL(651, 551)
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


class BezierCurveSurface(QMainWindow):
    def __init__(self, *args):
        super(BezierCurveSurface, self).__init__(*args)
        loadUi('comp_graph/minimal.ui', self)
        self.__type_to_index = {}
        self.__index_to_draw = {}
        self.__type = 0
        self.__step_t = 0.01
        self.__step_u_v = 0.05
        self.__abs_value = 25
        self.__data = []
        self.__last_type = -1
        self.__rotate_press = False
        self.__scale_press = False
        self.__start_rotate_mouse = None
        self.__start_scale_mouse = None
        self.__rotate_axis = None
        self.__rotate_angle = 0
        self.__scale_weight = 0
        self.__color_curve = (1, 0, 0)
        self.__color_point = (0, 1, 0)
        self.__delta_rotate_angle = 0
        self.__delta_scale = 0
        self.__points_general = 5
        self.__curves_general = 1

    def setupUI(self):
        self.openGLWidget.initializeGL()
        self.openGLWidget.resizeGL(651,551)
        self.openGLWidget.paintGL = self.draw

        curve = self.menuBar().addMenu('Curve')
        surface = self.menuBar().addMenu('Surface')

        def callback(title):
            def wrapper():
                self.__type = self.__type_to_index[title]
                self.openGLWidget.update()
            return wrapper

        type_number = 1
        for title, draw in [('Linear', self.bezier_curve_linear), ('Quadratic', self.bezier_curve_quadratic),
                            ('Cubic', self.bezier_curve_cubic), ('Linear', self.bezier_surface_linear),
                            ('Quadratic', self.bezier_surface_quadratic), ('Cubic', self.bezier_surface_cubic)]:
            curve_type = QAction(title, self)
            key_title = ''
            if type_number <= 3:
                key_title = curve.title()+curve_type.text()
            else:
                key_title = surface.title()+curve_type.text()
            curve_type.triggered.connect(callback(key_title))
            self.__type_to_index[key_title] = type_number
            self.__index_to_draw[type_number] = draw
            if type_number <= 3:
                curve.addAction(curve_type)
            else:
                surface.addAction(curve_type)
            type_number += 1

        curve_type = QAction('General', self)
        def general_curve():
            dialog = QDialog(self)
            form = QFormLayout(dialog)
            number_points = QSpinBox(dialog)
            number_points.setMinimum(5)
            form.addRow('Number points', number_points)
            button = QPushButton('Plot')
            form.addRow(button)
            def check():
                self.__points_general = number_points.value()
                self.__curves_general = 1
                self.__type = self.__type_to_index['CurveGeneral']
                self.openGLWidget.update()
                dialog.close()
            button.clicked.connect(check)
            dialog.show()
        curve_type.triggered.connect(general_curve)
        curve.addAction(curve_type)
        self.__type_to_index['CurveGeneral'] = type_number
        self.__index_to_draw[type_number] = self.bezier_surface_general
        type_number += 1

        surface_type = QAction('General', self)
        def general_surface():
            dialog = QDialog(self)
            form = QFormLayout(dialog)
            number_points = QSpinBox(dialog)
            number_points.setMinimum(5)
            form.addRow('Number points', number_points)
            number_curves = QSpinBox(dialog)
            number_curves.setMinimum(5)
            form.addRow('Number curves', number_curves)
            button = QPushButton('Plot')
            form.addRow(button)

            def check():
                self.__points_general = number_points.value()
                self.__curves_general = number_curves.value()
                self.__type = self.__type_to_index['SurfaceGeneral']
                self.openGLWidget.update()
                dialog.close()

            button.clicked.connect(check)
            dialog.show()

        surface_type.triggered.connect(general_surface)
        surface.addAction(surface_type)
        self.__type_to_index['SurfaceGeneral'] = type_number
        self.__index_to_draw[type_number] = self.bezier_surface_general



    def draw(self):
        if self.__type not in self.__index_to_draw:
            return
        self.__index_to_draw[self.__type]()

    def bezier_curve_linear(self):
        self.bezier_surface_general(1, 2)

    def bezier_curve_quadratic(self):
        self.bezier_surface_general(1, 3)

    def bezier_curve_cubic(self):
        self.bezier_surface_general(1, 4)

    def generate_layer(self, from_y, to_y, number_points):
        layer = []
        step_z = 2*self.__abs_value/number_points
        z = -self.__abs_value
        for _ in range(number_points):
            layer.append((random.uniform(-self.__abs_value, self.__abs_value), random.uniform(from_y, to_y),
                          random.uniform(z, z+step_z)))
            z += step_z
        return layer

    def bezier_surface_linear(self):
        self.bezier_surface_general(2, 2)

    def bezier_surface_quadratic(self):
        self.bezier_surface_general(3, 3)

    def bezier_surface_cubic(self):
        self.bezier_surface_general(4, 4)

    def bezier_surface_general(self, number_curves=-1, number_points=-1):
        if number_points == -1:
            number_points = self.__points_general
        if number_curves == -1:
            number_curves = self.__curves_general

        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        if self.__last_type != self.__type:
            self.__data = []
            for _ in range(number_curves):
                self.__data.append(self.generate_layer(-self.__abs_value, self.__abs_value, number_points))
            if glGetInteger(GL_MODELVIEW_STACK_DEPTH) > 1:
                glPopMatrix()

        if glGetInteger(GL_MODELVIEW_STACK_DEPTH) > 1:
            glPopMatrix()
        else:
            glLoadIdentity()

        if self.__rotate_press:
            glRotatef(self.__delta_rotate_angle, self.__rotate_axis[0], -self.__rotate_axis[1], 0.0)
        if self.__scale_press:
            glScalef(self.__delta_scale, self.__delta_scale, self.__delta_scale)

        glColor3f(self.__color_curve[0], self.__color_curve[1], self.__color_curve[2])
        u, v = 0.0, 0.0
        n = len(self.__data)
        m = len(self.__data[0])
        while u < 1.0 or abs(u - 1) < 1e-6:
            v = 0.0
            glBegin(GL_LINE_STRIP)
            while v < 1.0 or abs(v - 1) < 1e-6:
                point = [0, 0, 0]
                for i in range(n):
                    for j in range(m):
                        for c in range(3):
                            point[c] += self.__data[i][j][c] * math.comb(n - 1, i) * (u ** i) * ((1 - u) ** (n - 1 - i)) * \
                                        math.comb(m - 1, j) * (v ** j) * ((1 - v) ** (m - 1 - j))

                v += self.__step_u_v
                glVertex3f(point[0], point[1], point[2])
            glEnd()
            u += self.__step_u_v
        glPointSize(5)
        glColor3f(self.__color_point[0], self.__color_point[1], self.__color_point[2])
        glBegin(GL_POINTS)
        for i in range(n):
            for j in range(m):
                glVertex3f(self.__data[i][j][0], self.__data[i][j][1], self.__data[i][j][2])
        glEnd()

        if glGetInteger(GL_MODELVIEW_STACK_DEPTH) < 2:
            glPushMatrix()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-self.__abs_value * 3, self.__abs_value * 3, -self.__abs_value * 3, self.__abs_value * 3,
                -self.__abs_value * 2, self.__abs_value * 2)
        if self.__last_type == -1:
            self.__last_type = self.__type
            self.openGLWidget.update()
        else:
            self.__last_type = self.__type

    def mousePressEvent(self, event):
        pos = event.globalPos()
        if event.button() == Qt.LeftButton:
            self.__rotate_press = True
            self.__start_rotate_mouse = (pos.x(), pos.y())
            self.__rotate_angle = 0
        elif event.button() == Qt.RightButton:
            self.__scale_press = True
            self.__scale_weight = 0
            self.__start_scale_mouse = (pos.x(), pos.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__rotate_press = False
        elif event.button() == Qt.RightButton:
            self.__scale_press = False

    def mouseMoveEvent(self, event):
        pos = event.globalPos()
        def distance(a):
            return math.sqrt((a[0]-pos.x())**2+(a[1]-pos.y())**2)
        def normal(a):
            dist = distance(a)
            angle = math.atan2(pos.y()-a[1], pos.x()-a[0])+math.pi/2
            return (dist*math.cos(angle), dist*math.sin(angle))
        if self.__rotate_press:
            self.__rotate_axis = normal(self.__start_rotate_mouse)
            angle = distance(self.__start_rotate_mouse)
            self.__delta_rotate_angle = angle-self.__rotate_angle
            self.__rotate_angle = angle
            self.openGLWidget.update()
        if self.__scale_press:
            scale = min(distance(self.__start_scale_mouse), 200)/200
            self.__delta_scale = 1 + (scale - self.__scale_weight) * (-1 if pos.x() < self.__start_scale_mouse[0] else 1)
            self.__scale_weight = scale
            self.openGLWidget.update()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BezierCurveSurface()
    window.setupUI()
    window.show()
    sys.exit(app.exec_())
