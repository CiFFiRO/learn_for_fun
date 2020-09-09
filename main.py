import matplotlib.pyplot
import numpy as np
import cv2
import math


def image_open(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def images_show(images):
    for image in images:
        cv2.imshow(image[1], cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def grayworld(image, p=None):
    balance = cv2.xphoto.createGrayworldWB()
    if p is not None:
        balance.setSaturationThreshold(p)
    return balance.balanceWhite(image)


def perfect_reflector(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    max_value = -1
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            max_value = max(image_hsv[i][j][2], max_value)

    max_value = max(1, max_value)
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            image_hsv[i][j][2] = image_hsv[i][j][2] * 255 / max_value
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def histograms(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    histogram = [list(x) for x in cv2.calcHist([grey_image], [0], None, [256], [0, 256])]
    for k in range(3):
        rgb_histogram = cv2.calcHist([image], [k], None, [256], [0, 256])
        for i in range(256):
            histogram[i].append(rgb_histogram[i])
    return histogram


def plot_histogram(histograms, channels=None):
    rows = len(histograms) // 2 + (1 if len(histograms) % 2 == 1 else 0)
    cols = 1 if len(histograms) == 1 else 2
    colors = ['#808080', 'r', 'g', 'b']
    all_channels = ['grey', 'r', 'g', 'b']

    fig, ax = matplotlib.pyplot.subplots(nrows=rows, ncols=cols)
    for index in range(len(ax)):
        for i in range(4):
            if channels is None or all_channels[i] in channels:
                ax[index].plot(np.arange(0, 256), np.array([x[i] for x in histograms[index][0]]), color=colors[i])
        ax[index].set_facecolor('white')
        ax[index].set_xlim([0, 255])
        ax[index].set_xlabel('Value pixel')
        ax[index].set_ylabel('Quantity pixels')
        ax[index].axhline(y=0, color='k')
        ax[index].set_title(histograms[index][1])
        index += 1
    fig.canvas.set_window_title('Histogram')
    fig.set_facecolor('white')

    matplotlib.pyplot.show()


def linear_correction(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    max_value = -1
    min_value = 300
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            max_value = max(image_hsv[i][j][2], max_value)
            min_value = min(image_hsv[i][j][2], min_value)

    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            image_hsv[i][j][2] = 255 * (image_hsv[i][j][2] - min_value) / (max_value - min_value)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def gamma_correction(image, gamma):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            image_hsv[i][j][2] = math.floor(255 * (image_hsv[i][j][2]/255)**gamma)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def logarithmic_correction(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    max_value = -1
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            max_value = max(image_hsv[i][j][2], max_value)
    c = 255 / math.log(1.0+max_value)
    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            image_hsv[i][j][2] = math.floor(c * math.log(1.0+image_hsv[i][j][2]))
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def color_correction(image):
    result = image.copy()
    max_value = [-1, -1, -1]
    min_value = [300, 300, 300]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                max_value[k] = max(image[i][j][k], max_value[k])
                min_value[k] = min(image[i][j][k], min_value[k])

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(3):
                result[i][j][k] = 255 * (image[i][j][k] - min_value[k]) / (max_value[k] - min_value[k])
    return result


def clear_noise_salt_and_pepper(image):
    return cv2.medianBlur(image, 3)


def clear_noise_gauss(image):
    return cv2.GaussianBlur(image, (3, 3), 3)


def up_sharpness(image, alpha):
    gaussian_kernel = cv2.getGaussianKernel(ksize=3, sigma=3)*cv2.getGaussianKernel(ksize=3, sigma=3).transpose()
    kernel = (1.0+alpha)*np.array([[0,0,0], [0,1,0], [0,0,0]])-alpha*gaussian_kernel
    return cv2.filter2D(image, -1, kernel)


def multi_scale_retinex(image, ksize=0, sigma=9):
    I = image.copy()
    I = cv2.convertScaleAbs(np.float32(I), I, alpha=1.0, beta=1.0)
    I = cv2.log(np.float32(I), I)

    l = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    l = cv2.convertScaleAbs(np.float32(l), l, alpha=1.0, beta=1.0)
    l = cv2.log(np.float32(l), l)

    result = cv2.subtract(I, l, l)
    result = cv2.exp(result, result)
    frames = cv2.split(result)
    for i in range(3):
        frames[i] = cv2.normalize(frames[i], frames[i], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        frames[i] = cv2.convertScaleAbs(frames[i], frames[i], alpha=1/3, beta=0.0)
    return color_correction(cv2.merge(frames))


def MSE(image_1, image_2):
    if image_1.shape[0] != image_2.shape[0] and image_1.shape[1] != image_2.shape[1]:
        raise AssertionError('Different dimensions images')

    a = np.float32(cv2.cvtColor(image_1, cv2.COLOR_RGB2HSV))
    b = np.float32(cv2.cvtColor(image_2, cv2.COLOR_RGB2HSV))
    result = 0.0
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            result += ((a[i][j][2]-b[i][j][2])**2)
    return result/(a.shape[0]*a.shape[1])


def PSNR(image_1, image_2):
    mse = MSE(image_1, image_2)
    if abs(mse) < 1e-12:
        return 0.0
    return 10*(2*math.log10(255)-math.log10(MSE(image_1, image_2)))


if __name__ == '__main__':
    image = image_open('image/dif_light_2.jpg')



