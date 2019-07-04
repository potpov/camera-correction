import conf
import cv2
import numpy as np


def new_axes(xc, yc, xd, yd, k1):
    r2 = (xd ** 2 + yd ** 2)
    dist = k1 * r2
    xu = int(xd + (xd - xc) * dist)
    yu = int(yd + (yd - yc) * dist)
    return xu, yu


def inv_axes(xc, yc, xu, yu, k1):
    ru = np.sqrt(xu ** 2 + yu ** 2)
    coeff1 = ru / (2 * k1)
    coeff2 = (1 / (3 * k1)) ** 3
    coeff3 = (ru / (2 * k1)) ** 2
    rd = np.cbrt(coeff1 + np.sqrt(coeff2 + coeff3)) + np.cbrt(coeff1 - np.sqrt(coeff2 + coeff3))
    xd = int(xc + (xu - xc) * (rd / ru))
    yd = int(yc + (yu - yc) * (rd / ru))
    return xd, yd


def apply_distortion(img, k1):
    """
    apply distortion over an image with the parameter k1
    :param img: input numpy array
    :param k1: distortion coefficient
    :return:
    """
    width = img.shape[1]
    height = img.shape[0]

    corr_x, corr_y = new_axes(0, 0, width // 2, height // 2, k1)

    maxxu = int(corr_x)
    maxyu = int(corr_y)

    minytop = 0
    minxlef = 0
    minybot = 2 * maxyu
    minxrig = 2 * maxxu

    out_mat = np.zeros((2 * maxyu, 2 * maxxu, len(img.shape)), dtype="uint8")
    # if len(img.shape) == 3:
    #     out_mat = np.zeros((2 * maxyu, 2 * maxxu, 3), dtype="uint8")
    # else:
    #     out_mat = np.zeros((2 * maxyu, 2 * maxxu), dtype="uint8")

    for y_iter in range(2 * maxyu):
        for x_iter in range(2 * maxxu):

            xd = - maxxu + x_iter
            yd = - maxyu + y_iter

            if (xd == 0) and (yd == 0):
                corr_x, corr_y = 0, 0
            else:
                corr_x, corr_y = inv_axes(0, 0, xd, yd, k1)

            matxu = corr_x + width // 2
            matyu = corr_y + height // 2

            if len(img.shape) == 3:
                if matxu >= width:
                    out_mat[y_iter, x_iter, :] = 0
                elif matyu >= height:
                    out_mat[y_iter, x_iter, :] = 0
                elif matxu <= 0:
                    out_mat[y_iter, x_iter, :] = 0
                elif matyu <= 0:
                    out_mat[y_iter, x_iter, :] = 0
                else:
                    out_mat[y_iter, x_iter, np.newaxis] = img[matyu, matxu, :]
            else:
                if matxu >= width:
                    out_mat[y_iter, x_iter] = 0
                elif matyu >= height:
                    out_mat[y_iter, x_iter] = 0
                elif matxu <= 0:
                    out_mat[y_iter, x_iter] = 0
                elif matyu <= 0:
                    out_mat[y_iter, x_iter] = 0
                else:
                    out_mat[y_iter, x_iter] = img[matyu, matxu]

            if matyu == 0 and y_iter > minytop:
                minytop = y_iter
            elif matyu == height and y_iter < minybot:
                minybot = y_iter
            elif matxu == 0 and x_iter > minxlef:
                minxlef = x_iter
            elif matxu == width and x_iter < minxrig:
                minxrig = x_iter

    out_mat = np.uint8(out_mat[minytop:minybot, minxlef:minxrig])

    return out_mat, minytop, minybot, minxlef, minxrig


def sfrs_calibrate(img):
        """
        loop over epoch applying distortion with different coefficients,
        scoring the results and selecting the best or the input image if none
        of the used coefficients fit this input
        :param img: numpy array
        :return: numpy array
        """
        rsz = cv2.resize(img, (0, 0), fx=.6, fy=.6)
        k = conf.CAL_START_COEFF
        res = {}
        add_k = conf.CAL_COEFF_INC_FACTOR

        for i in range(conf.CAL_EPOCH_NUMB):
            # apply distortion with the current coeff k
            c, _, _, _, _ = apply_distortion(rsz, k)

            # get straight lines in the new image using hough
            gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 110)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)

            # evaluating distortion using number of lines
            if lines is not None:
                res[k] = len(lines)
            else:
                res[k] = 0

            print("epoch: ", i, "k: ", float(k), "score: ", res[k])
            k += add_k

        best = max(res, key=res.get)

        print("the winner is: ")
        print("epoch: ", float(best), "score: ", res[best])
        if float(best) != 0:
            output, _, _, _, _ = apply_distortion(rsz, float(best))
        else:
            output = rsz

        return output
