import matplotlib.pyplot
import numpy as np
import cv2
import math
import numba
import time
import random
from sklearn.neighbors import KDTree
import skimage


HISTOGRAM_CHANNEL_COLOR_GREY = '#808080'
HISTOGRAM_CHANNEL_COLOR_RED = 'r'
HISTOGRAM_CHANNEL_COLOR_GREEN = 'g'
HISTOGRAM_CHANNEL_COLOR_BLUE = 'b'
COMPARISON_PATTERN_METRIC_SAD = 'SAD'
COMPARISON_PATTERN_METRIC_SSD = 'SSD'
COMPARISON_PATTERN_METRIC_CC = 'CC'
COMPARISON_PATTERN_LIGHT_RETINEX = 'RETINEX'
COMPARISON_PATTERN_LIGHT_NORM = 'NORM'
COMPARISON_PATTERN_FILTER_MED = 'MED'
COMPARISON_PATTERN_FILTER_GAUSS = 'GAUSS'
FEATURE_TYPE_BLOB = 'BLOB'
FEATURE_TYPE_CORNER = 'CORNER'
FEATURE_TYPE_REGION = 'REGION'


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

    @numba.njit
    def correction(image_hsv):
        max_value = -1
        min_value = 300
        for i in range(image_hsv.shape[0]):
            for j in range(image_hsv.shape[1]):
                max_value = max(image_hsv[i][j][2], max_value)
                min_value = min(image_hsv[i][j][2], min_value)

        for i in range(image_hsv.shape[0]):
            for j in range(image_hsv.shape[1]):
                image_hsv[i][j][2] = 255 * (image_hsv[i][j][2] - min_value) / (max_value - min_value)
    correction(image_hsv)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def gamma_correction(image, gamma):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    @numba.njit
    def correction(image_hsv):
        for i in range(image_hsv.shape[0]):
            for j in range(image_hsv.shape[1]):
                image_hsv[i][j][2] = math.floor(255 * ((image_hsv[i][j][2]+.0)/255)**gamma)
    correction(image_hsv)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def logarithmic_correction(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    @numba.njit
    def correction(image_hsv):
        max_value = -1
        for i in range(image_hsv.shape[0]):
            for j in range(image_hsv.shape[1]):
                max_value = max(image_hsv[i][j][2], max_value)
        c = 255 / math.log(1.0+max_value)
        for i in range(image_hsv.shape[0]):
            for j in range(image_hsv.shape[1]):
                image_hsv[i][j][2] = math.floor(c * math.log(1.0+image_hsv[i][j][2]))

    correction(image_hsv)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def color_correction(image):
    result = image.copy()
    max_value = np.array([-1, -1, -1])
    min_value = np.array([300, 300, 300])

    @numba.njit
    def correction(result, image, max_value, min_value):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):
                    max_value[k] = max(image[i][j][k], max_value[k])
                    min_value[k] = min(image[i][j][k], min_value[k])

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                for k in range(3):
                    result[i][j][k] = 255 * (image[i][j][k] - min_value[k]) / (max_value[k] - min_value[k])
    correction(result, image, max_value, min_value)
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

    @numba.njit
    def mse(a, b):
        result = 0.0
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                result += ((a[i][j][2]-b[i][j][2])**2)
        return result
    return mse(a, b)/(a.shape[0]*a.shape[1])


def PSNR(image_1, image_2):
    mse = MSE(image_1, image_2)
    if abs(mse) < 1e-12:
        return 0.0
    return 10*(2*math.log10(255)-math.log10(MSE(image_1, image_2)))


def naive_comparison_pattern(in_image, in_pattern, precision, light=COMPARISON_PATTERN_LIGHT_NORM,
                             metric=COMPARISON_PATTERN_METRIC_SAD):
    @numba.njit
    def light_pixel_norm(image):
        I_avr = 0.0
        I_norm = 0.0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                I_norm += (image[i][j][2] - I_avr) ** 2
                I_avr += image[i][j][2]
        I_avr /= image.shape[0] * image.shape[1]
        I_norm = math.sqrt(I_norm)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j][2] = (image[i][j][2] - I_avr) / I_norm

    if in_pattern.shape[0] > in_image.shape[0] or in_pattern.shape[1] > in_image.shape[1]:
        return None

    image = in_image.copy()
    pattern = in_pattern.copy()
    result = None
    if light == COMPARISON_PATTERN_LIGHT_NORM:
        image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2HSV)
        pattern = cv2.cvtColor(np.float32(pattern), cv2.COLOR_RGB2HSV)
        light_pixel_norm(image)
        light_pixel_norm(pattern)
    else:
        image = cv2.cvtColor(multi_scale_retinex(image), cv2.COLOR_RGB2HSV)
        pattern = cv2.cvtColor(multi_scale_retinex(pattern), cv2.COLOR_RGB2HSV)


    image = np.float32(image)
    pattern = np.float32(pattern)

    @numba.njit
    def dist(x, y):
        distance = 0.0
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if metric == COMPARISON_PATTERN_METRIC_SAD:
                    distance += abs(image[x+i][y+j][2] - pattern[i][j][2])
                elif metric == COMPARISON_PATTERN_METRIC_SSD:
                    distance += abs(image[x + i][y + j][2] - pattern[i][j][2])**2
                else:
                    distance += image[x + i][y + j][2]*pattern[i][j][2]
        return distance

    value = np.float32([0])
    coordinates = np.float32([0, 0])
    for i in range(0, image.shape[0] - pattern.shape[0]):
        for j in range(0, image.shape[1] - pattern.shape[1]):
            distance = dist(i, j)
            if i == 0 and j == 0:
                value = distance
                coordinates = (i, j)
            else:
                if metric == COMPARISON_PATTERN_METRIC_CC:
                    if value < distance:
                        coordinates = (i, j)
                        value = distance
                else:
                    if value > distance:
                        coordinates = (i, j)
                        value = distance

    print(value)
    if (metric == COMPARISON_PATTERN_METRIC_CC and value > precision) or \
            (metric != COMPARISON_PATTERN_METRIC_CC and value < precision):
        result = coordinates

    return result


def mark_enterings(image, entering, color=(255,0,0)):
    result = image.copy()
    for pattern, (x, y) in entering:
        for j in range(pattern.shape[0]):
            result[x + j][y] = color
            result[x + j][y + pattern.shape[1] - 1] = color
        for j in range(pattern.shape[1]):
            result[x][y + j] = color
            result[x + pattern.shape[0] - 1][y + j] = color
    return result


def edge_detector(image, link=255 / 3, find=255):
    return cv2.Canny(image, find, link)


def edges_comparison_pattern(in_image, in_pattern, filter=COMPARISON_PATTERN_FILTER_MED, precision=None):
    image = in_image.copy()
    pattern = in_pattern.copy()
    if filter == COMPARISON_PATTERN_FILTER_MED:
        image = cv2.medianBlur(image, 3)
        pattern = cv2.medianBlur(pattern, 3)
    elif filter == COMPARISON_PATTERN_FILTER_GAUSS:
        image = cv2.GaussianBlur(image, (3,3), 3)
        pattern = cv2.GaussianBlur(pattern, (3,3), 3)
    image = edge_detector(multi_scale_retinex(image))
    pattern = edge_detector(multi_scale_retinex(pattern))

    @numba.njit
    def inverse_edge_value(image):
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i][j] > 0:
                    image[i][j] = 0.0
                else:
                    image[i][j] = 255.0

    inverse_edge_value(image)
    image = cv2.distanceTransform(image, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)

    @numba.njit
    def dist(image, pattern, x, y):
        distance = 0.0
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if pattern[i][j] > 0:
                    distance += image[x+i][y+j]
        return distance

    value = None
    coordinates = None
    for i in range(0, image.shape[0] - pattern.shape[0]):
        for j in range(0, image.shape[1] - pattern.shape[1]):
            distance = dist(image, pattern, i, j)
            if value is None or value > distance:
                value = distance
                coordinates = (i, j)

    if precision is not None:
        if value < precision:
            return coordinates
        else:
            return None

    return coordinates


def binary_to_grey_image(image):
    result = image.copy()
    @numba.njit
    def convert(image, result):
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i][j] < 1:
                    result[i][j] = 0
                else:
                    result[i][j] = 255
    convert(image, result)
    return result


def threshold_binarization(in_image, threshold, dark_background):
    if dark_background:
        return cv2.threshold( cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY), threshold,1, cv2.THRESH_BINARY)[1]
    return cv2.threshold(cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY), threshold, 1, cv2.THRESH_BINARY_INV)[1]


def threshold_binarization_by_histogramm(image, dark_background, p=0.05):
    grey = [x[0] for x in histograms(image)]
    for i in range(1, len(grey)-1):
        grey[i] = (grey[i-1]+grey[i]+grey[i+1])/3
    value = grey[0]
    h_max = 0
    for i in range(1, len(grey)):
        if value < grey[i]:
            value = grey[i]
            h_max = i
    index = h_max
    all_pixels = 0
    while (dark_background and index >= 0) or (not dark_background and index < len(grey)):
        all_pixels += grey[index]
        if dark_background:
            index -= 1
        else:
            index += 1
    index = 0 if dark_background else len(grey)-1
    p_pixels = 0
    while index != h_max and p_pixels/all_pixels < p:
        p_pixels += grey[index]
        if dark_background:
            index += 1
        else:
            index -= 1
    if dark_background:
        index += 1
    else:
        index -= 1
    return threshold_binarization(image, 2*h_max-index, dark_background)


def adaptation_binarization(in_image, r, c, dark_background):
    if dark_background:
        return cv2.adaptiveThreshold(cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY), 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, r, c)
    return cv2.adaptiveThreshold(cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY), 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, r, c)


def mat_constriction(image, ksize):
    return cv2.erode(image, np.ones((ksize, ksize)))


def mat_expansion(image, ksize):
    return cv2.dilate(image, np.ones((ksize, ksize)))


def mat_open(image, ksize):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((ksize, ksize)))


def mat_close(image, ksize):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((ksize, ksize)))


def select_connected_components(image, connectivity=4):
    return cv2.connectedComponents(image, connectivity)


def coloring_components(image, number_components):
    result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    colors = np.zeros((number_components, 3), dtype=np.uint8)
    for i in range(1, number_components):
        for j in range(3):
            colors[i][j] = random.randint(25, 230)

    @numba.njit
    def coloring(image, result, colors):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] > 0:
                    for k in range(3):
                        result[i][j][k] = colors[image[i][j]][k]

    coloring(image, result, colors)
    return result


# (perimeter, area, compact, elongation, orientation_leader_axis)
def geometric_properties_components(image_labled, binary_image, number_components):
    boundaries = mat_constriction(binary_image, 3)
    perimeter = np.zeros(number_components, dtype=np.float32)
    area = np.zeros(number_components, dtype=np.float32)
    center_mass = np.zeros((number_components, 2), dtype=np.float32)
    m = np.zeros((number_components, 3, 3), dtype=np.float32)
    elongation = np.zeros(number_components, dtype=np.float32)
    orientation_leader_axis = np.zeros(number_components, dtype=np.float32)

    @numba.njit
    def calculate_boundaries(boundaries, binary_image):
        for i in range(boundaries.shape[0]):
            for j in range(boundaries.shape[1]):
                boundaries[i][j] = binary_image[i][j]-boundaries[i][j]

    @numba.njit
    def first_calculate(image_labled, boundaries, perimeter, area, center_mass):
        for i in range(boundaries.shape[0]):
            for j in range(boundaries.shape[1]):
                if image_labled[i][j] > 0:
                    area[image_labled[i][j]] += 1
                    center_mass[image_labled[i][j]][0] += i
                    center_mass[image_labled[i][j]][1] += j
                if boundaries[i][j] > 0:
                    perimeter[image_labled[i][j]] += 1

    @numba.njit
    def second_calculate(image_labled, center_mass, m):
        for i in range(image_labled.shape[0]):
            for j in range(image_labled.shape[1]):
                if image_labled[i][j] > 0:
                    for k in range(3):
                        for l in range(3):
                            m[image_labled[i][j]][k][l] += (center_mass[image_labled[i][j]][0] - i) ** k * \
                                                           (center_mass[image_labled[i][j]][1] - j) ** l

    calculate_boundaries(boundaries, binary_image)
    first_calculate(image_labled, boundaries, perimeter, area, center_mass)

    for i in range(1, number_components):
        center_mass[i][0] /= area[i]
        center_mass[i][1] /= area[i]
    second_calculate(image_labled, center_mass, m)
    for i in range(1, number_components):
        elongation[i] = (m[i][2][0] + m[i][0][2] + np.sqrt((m[i][2][0] - m[i][0][2])**2 + 4*m[i][1][1]**2)) / \
                        (m[i][2][0] + m[i][0][2] - np.sqrt((m[i][2][0] - m[i][0][2]) ** 2 + 4 * m[i][1][1] ** 2))
        orientation_leader_axis[i] = 0.5*np.arctan(2*m[i][1][1]/(m[i][2][0]-m[i][0][2]))
    result = []
    for i in range(1, number_components):
        result.append((i, perimeter[i], area[i], perimeter[i]**2/area[i], elongation[i], orientation_leader_axis[i]))

    return result


def photometric_properties_components(image_labled, origin_image, number_components):
    image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2HSV)
    historam_component = np.zeros((number_components, 256), dtype=np.float32)
    avr_value = np.zeros(number_components, dtype=np.float32)
    area = np.zeros(number_components, dtype=np.float32)
    mat_delay = np.zeros(number_components, dtype=np.float32)
    dispersion = np.zeros(number_components, dtype=np.float32)

    @numba.njit
    def first_calculate(image_labled, image, avr_value, area, historam_component):
        for i in range(image_labled.shape[0]):
            for j in range(image_labled.shape[1]):
                if image_labled[i][j] > 0:
                    avr_value[image_labled[i][j]] += image[i][j][2]
                    area[image_labled[i][j]] += 1
                    historam_component[image_labled[i][j]][image[i][j][2]] += 1

    @numba.njit
    def second_calculate(image_labled, image, mat_delay, dispersion, historam_component):
        for i in range(image_labled.shape[0]):
            for j in range(image_labled.shape[1]):
                if image_labled[i][j] > 0:
                    dispersion[image_labled[i][j]] += (image[i][j][2]-mat_delay[image_labled[i][j]])**2 * \
                                                      historam_component[image_labled[i][j]][image[i][j][2]]

    first_calculate(image_labled, image, avr_value, area, historam_component)
    for i in range(1, number_components):
        avr_value[i] /= area[i]
        for j in range(256):
            mat_delay[i] += historam_component[i][j]*j
        historam_component[i] /= area[i]
        mat_delay[i] /= area[i]
    second_calculate(image_labled, image, mat_delay, dispersion, historam_component)
    result = []
    for i in range(1, number_components):
        result.append((i, avr_value[i], dispersion[i], historam_component[i]))
    return result


def mark_features(image, features, type, color=(255,0,0)):
    result = image.copy()
    for feature in features:
        if type == FEATURE_TYPE_BLOB:
            x, y, r = feature
            cv2.circle(result, (round(x), round(y)), round(r), color)
        elif type == FEATURE_TYPE_CORNER:
            x, y = feature
            cv2.circle(result, (round(x), round(y)), 1, color)
        else:
            x, y, width, height = feature
            cv2.rectangle(result, (x, y), (min(x+width, result.shape[0]-1), min(y+height, result.shape[1]-1)), color)

    return result


def blob_detector(image, threshold=0.5):
    # SimpleBlobDetector detector in opencv
    result = skimage.feature.blob_dog(cv2.cvtColor(clear_noise_gauss(image), cv2.COLOR_RGB2GRAY), threshold=threshold)
    result[:, 2] = result[:, 2] * np.sqrt(2.0)
    return [[x[1], x[0], x[2]] for x in result]


def corner_detector(image, threshold_corn=0.01, threshold_dog=0.01, number=5000):
    detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(corn_thresh=threshold_corn, DOG_thresh=threshold_dog,
                                                                   maxCorners=number)
    result = detector.detect(cv2.cvtColor(clear_noise_gauss(image), cv2.COLOR_RGB2GRAY))
    return [x.pt for x in result]


def region_detector(image):
    detector = cv2.MSER_create()
    _, result = detector.detectRegions(clear_noise_gauss(image))
    return result


# O(n r^2 log n), may be other method decrease complexity to O(n log n), n=len(features_in)
def adaptive_select_feature(features_in, type, radius):
    result = []
    used_points = set()
    features = list(features_in.copy())
    if type == FEATURE_TYPE_REGION:
        features.sort(key=lambda x: x[2]*x[3], reverse=True)
    for feature in features:
        x, y  = None, None
        if type == FEATURE_TYPE_BLOB:
            x, y, _ = feature
        elif type == FEATURE_TYPE_CORNER:
            x, y = feature
        else:
            x, y, w, h = feature
            x += round(w/2)
            y += round(h/2)

        if (x, y) not in used_points:
            result.append(feature)
            for i in range(2*radius+1):
                for j in range(2*radius+1):
                    used_points.add((x-radius+i, y-radius+j))
    return result


# (x, y, r), angle, descriptor
def SIFT(image):
    detector = cv2.SIFT_create()
    features, descriptors = detector.detectAndCompute(cv2.cvtColor(clear_noise_gauss(image), cv2.COLOR_RGB2GRAY), None)
    return [(round(x.pt[0]), round(x.pt[1]), x.size/2) for x in features], [x.angle for x in features], descriptors


def comparison_features(descriptors_1, descriptors_2, threshold=None):
    tree = KDTree(descriptors_2, metric='l1')
    points = [x[0] for x in tree.query(descriptors_1, return_distance=False)]

    if threshold is not None:
        def l1(a, b):
            result = 0
            for i in range(len(a)):
                result += abs(a[i]-b[i])
            return result
        for i in range(len(points)):
            if l1(descriptors_1[i], descriptors_2[points[i]]) > threshold:
                points[i] = None
    result = []
    for i in range(len(points)):
        if points[i] is not None:
            result.append((i, points[i]))
    return result


def show_pairs_features(image_1, image_2, pairs, features_1, features_2, type):
    a = mark_features(image_1, features_1, type)
    b = mark_features(image_2, features_2, type)

    result = np.concatenate((a, b), axis=1)
    for p_1, p_2 in pairs:
        x2, y2, x1, y1 = features_2[p_2][0], features_2[p_2][1], features_1[p_1][0], features_1[p_1][1]
        cv2.arrowedLine(result, (x1, y1), (image_1.shape[1]+x2, y2), (0, 0, 255))
    return result


def calculate_homography(kp1, kp2, pairs, T=2):
    points_1 = np.zeros((len(pairs), 1, 2), dtype=np.float32)
    points_2 = np.zeros((len(pairs), 1, 2), dtype=np.float32)
    for i in range(len(pairs)):
        points_1[i] = kp1[pairs[i][0]][0:2]
        points_2[i] = kp2[pairs[i][1]][0:2]
    homohraphy, _ = cv2.findHomography(points_1, points_2, cv2.RANSAC, ransacReprojThreshold=T)
    return homohraphy


def filter_pairs_features(features_1, features_2, pairs, transform, T):
    result = []
    M = np.array(transform)
    for p1, p2 in pairs:
        point1 = np.zeros((3, 1), dtype=np.float32)
        point2 = np.zeros((3, 1), dtype=np.float32)
        point1[0][0], point1[1][0], point1[2][0] = features_1[p1][0], features_1[p1][1], 1
        point2[0][0], point2[1][0], point2[2][0] = features_2[p2][0], features_2[p2][1], 1
        if not np.linalg.norm(point2 - M.dot(point1), ord='fro') > T:
            result.append((p1, p2))
    return result


def custom_create_panorama(image_1, image_2, H, offset_1, offset_2):
    width = image_1.shape[1] + image_2.shape[1]
    height = image_1.shape[0] + image_2.shape[0]
    offset = np.zeros((3,3), dtype=np.float32)
    for i in range(3):
        offset[i][i] = 1
    offset[0][2] = offset_1[0]
    offset[1][2] = offset_1[1]

    result = cv2.warpPerspective(image_1, H.dot(offset), (width, height))
    result[offset_2[1]:offset_2[1]+image_2.shape[0], offset_2[0]:offset_2[0]+image_2.shape[1]] = image_2
    return result


def create_panorama(images):
    stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    status, result = stitcher.stitch(images)
    if status != cv2.STITCHER_OK:
        return None
    return result


def hough_transform(image, rho=1.0, theta=np.pi/180, min_vote=100):
    edges = edge_detector(image)
    return [x[0] for x in cv2.HoughLinesP(edges, rho, theta, min_vote)]


def mark_lines(image, lines, color=(255, 0, 0)):
    result = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), color)
    return result


if __name__ == '__main__':
    image_1 = image_open('image/pan_0.jpg')
    image_2 = image_open('image/pan_1.jpg')
    s = time.time()
    kp1, _, descriptors1 = SIFT(image_1)
    kp2, _, descriptors2 = SIFT(image_2)
    pairs = comparison_features(descriptors1, descriptors2)
    h = calculate_homography(kp1, kp2, pairs)
    filt_pairs = filter_pairs_features(kp1, kp2, pairs, h, 4)
    e = time.time()
    print('time', e-s)
    # images_show([(show_pairs_features(image_1, image_2, pairs, kp1, kp2, FEATURE_TYPE_BLOB), 'image'),
    #              (show_pairs_features(image_1, image_2, filt_pairs, kp1, kp2, FEATURE_TYPE_BLOB), 'filt'),
    #              (cv2.warpPerspective(image_1, h, (image_1.shape[1], image_1.shape[0])), 'trans1'),
    #              (cv2.warpPerspective(image_2, np.linalg.inv(h), (image_2.shape[1], image_2.shape[0])), 'trans2')])
    images_show([(custom_create_panorama(image_1, image_2, h, (355, 0), (355, 0)), 'image')])

    # image = image_open('image/hough_trans.jpg')
    # lines_1 = hough_transform(image, rho=3, min_vote=10)
    # lines_2 = hough_transform(image, rho=3, min_vote=100)
    # images_show([(mark_lines(image, lines_1), '25'), (mark_lines(image, lines_2), '100')])
