#include <iostream>
#include <fstream>

// GLEW
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

const GLuint WIDTH = 800, HEIGHT = 600;

GLuint createShader(GLenum shaderType, const std::string& pathShaderFile) {
    auto readFile = [](const std::string& pathFile) -> std::string {
        std::ifstream file;
        file.open(pathFile, std::ios::in);
        std::string result { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
        file.close();
        return result;
    };

    std::string shaderCode = readFile(pathShaderFile);
    GLuint shader = glCreateShader(shaderType);
    const GLchar* text = shaderCode.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        const int lengthLog = 512;
        GLchar infoLog[lengthLog];
        glGetShaderInfoLog(shader, lengthLog, nullptr, infoLog);
        std::cerr << "ERROR compile shader: " << infoLog << '\n';
    }

    return shader;
}

class BaseWindow {
public:
    BaseWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

        window__ = glfwCreateWindow(WIDTH, HEIGHT, "", nullptr, nullptr);
        glfwMakeContextCurrent(window__);

        glewExperimental = GL_TRUE;
        glewInit();
    }

    void show() {
        while (!glfwWindowShouldClose(window__)) {
            loop();
        }
    }
    void initialize() {
        init();
        initGL();
    }

    virtual ~BaseWindow() {
        glfwTerminate();
    }
protected:
    GLFWwindow* window__;
private:

    virtual void init() = 0;
    virtual void initGL() = 0;
    virtual void loop() = 0;
};

class BorgShip : public BaseWindow {
public:
    BorgShip() : BaseWindow() {}

    ~BorgShip() override {
        glDeleteVertexArrays(1, &vertexArrayObject__);
        glDeleteBuffers(1, &vertexBufferObject__);
    }

private:
    int width__;
    int height__;
    GLuint shaderProgram__;
    GLuint vertexArrayObject__;
    GLuint vertexBufferObject__;

    void init() override {
        glfwSetKeyCallback(window__, key_callback);
        glfwGetFramebufferSize(window__, &width__, &height__);
    }

    void initGL() override {
        glViewport(0, 0, width__, height__);

        GLuint vertexShader = createShader(GL_VERTEX_SHADER, "..\\ship_vertex_shader.glsl");
        GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, "..\\ship_fragment_shader.glsl");

        shaderProgram__ = glCreateProgram();
        glAttachShader(shaderProgram__, vertexShader);
        glAttachShader(shaderProgram__, fragmentShader);
        glLinkProgram(shaderProgram__);
        int success;
        glGetProgramiv(shaderProgram__, GL_LINK_STATUS, &success);
        if (!success) {
            const int logSize = 512;
            GLchar infoLog[logSize];
            glGetProgramInfoLog(shaderProgram__, 512, nullptr, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);


        GLfloat vertices[] = {
                -0.5f, -0.5f, 0.0f,
                0.5f, -0.5f, 0.0f,
                0.0f,  0.5f, 0.0f
        };

        glGenVertexArrays(1, &vertexArrayObject__);
        glGenBuffers(1, &vertexBufferObject__);

        glBindVertexArray(vertexArrayObject__);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject__);

        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)nullptr);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);
    }

    void loop() override {
        glfwPollEvents();

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram__);
        glBindVertexArray(vertexArrayObject__);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        glfwSwapBuffers(window__);
    }
};

int main() {
    BorgShip* window = new BorgShip();
    window->initialize();
    window->show();
    delete window;

    return 0;
}

