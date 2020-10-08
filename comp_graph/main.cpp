#include <QWidget>
#include <QtCore>
#include <QtGui>
#include <QtOpenGL/QtOpenGL>

#include <QtOpenGL/QGLWidget>

#include <iostream>

class BorgShip : public QGLWidget {
public:
    explicit BorgShip(QWidget* parent = nullptr) : QGLWidget(parent) {
        resize(800, 600);
    }
protected:
    void initializeGL() override {

    }

    void resizeGL(int nWidth, int nHeight) override {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glViewport(0, 0, (GLint)nWidth, (GLint)nHeight);
    }

    void paintGL() override {

    }
};


int main(int argc, char** argv) {
    QApplication app(argc, argv);

    BorgShip* window = new BorgShip();
    window->setWindowTitle("BorgShip");
    window->show();

    return app.exec();
}

