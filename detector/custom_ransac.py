import numpy as np
import warnings
warnings.filterwarnings("ignore")


class RANSAC(object):
    """
    Class that implements the functionality of RANSAC algorithm.
    """
    def __init__(self):
        pass

    @classmethod
    def random_sample(cls, row_coords, col_coords, n):
        """
        Method that chooses a random n number of points
        and calculates a random sample consensus based on
        chosen coordinates (x and y).
        :param row_coords: x coordinates (row coordinates)
        :param col_coords: y coordinates (column coordinates)
        :param n: number of points to choose
        :return: x and y random points of the image
        """
        total_points = len(row_coords)
        # Forming random array from row coordinates
        random_x = np.random.permutation(np.arange(total_points))
        # Choosing random n numbers from random_x
        random_x = random_x[:n]
        # New coords for x and y to calculate the metric
        row_coords_new, col_coords_new = [], []
        for i in range(len(random_x)):
            row_coords_new.append(row_coords[random_x[i]])
            col_coords_new.append(col_coords[random_x[i]])
        return row_coords_new, col_coords_new

    # Calculation of line equation and solving them using matrix equation
    # A*P=q  -> P = A**(-1) * q
    @classmethod
    def find_line(cls, x1, y1, x2, y2):
        """
        Solving the equation of a line in matrix form.
        Algebraic form: y = ax + b
        Matrix Form: A * P = q.
        To solve in matrix form find: P = A**(-1) * q
        :param x1: first row coordinate
        :param x2: second row coordinate
        :param y1: first column coordinate
        :param y2: second column coordinate
        :return: a and b coordinates: numpy array.
        """
        p = np.array([1, 1])
        A = np.array([[x1, 1], [x2, 1]])
        q = np.array([[y1], [y2]])
        if np.linalg.det(A) != 0:
            p = np.linalg.inv(A).dot(q)
            return p[0], p[1]
        else:
            return p[0], p[1]

    # computing consistency score (simple algebraic distance between points) for each point on the given space
    @classmethod
    def get_consistency_score(cls, x_coords, y_coords, threshold, line_x, line_y):
        """
        Function that calculates the consistency score based on algebraic distance for
        any point on the given space.
        :param x_coords: x_coordinates for point(-s)
        :param y_coords: y_coordinates for point(-s)
        :param threshold: the parameter for controlling how close should be point to the given line
        :param line_x: x coordinates for the line
        :param line_y: y coordinates for the line
        :return: scalar value of consistency score
        """
        consistency_score = 0
        for i in range(len(x_coords)):
            squared_error = (line_x * x_coords[i] + line_y - y_coords[i]) ** 2
            if squared_error < threshold:
                consistency_score += 1
        return consistency_score

    # Finding the best line using RANSAC
    @classmethod
    def fit_RANSAC(cls, image, epochs=100, sigma=2):
        """
        RANSAC algorithm for finding the best line on a given image.
        :param sigma: closing distance to the line (e.g. closer will be narrower line)
        :param epochs: number of iterations to find the best line (default 100)
        :param image: array of edged image.
        :return: line coords in form of numpy array
        """
        # filtering from 0's
        image = image > 0
        # computing the coordinates for all edge pixels
        idx = np.argwhere(image)
        row_coordinates, column_coordinates = idx[:, 0], idx[:, 1]
        best_a, best_b = 0, 0
        # Compute the line with best score
        max_score = -1
        for iteration in range(epochs):
            row_coordinates_2, column_coordinates_2 = cls.random_sample(row_coordinates, column_coordinates, 2)
            a, b = cls.find_line(row_coordinates_2[0], column_coordinates_2[0], row_coordinates_2[1],
                             column_coordinates_2[1])
            # Computing the score
            score = cls.get_consistency_score(row_coordinates, column_coordinates, sigma, a, b)
            if score > max_score:
                max_score = score
                best_a, best_b = a, b
        return best_a, best_b

    # Function to extend found line on the whole image
    @classmethod
    def generate_extends(cls, a_coords, b_coords, img_rows, img_cols):
        """
        Function to draw (extend) found line for the whole image.
        :param a_coords: a matrix coordinates for best fitted line.
        :param b_coords: b matrix coordinates for best fitted line
        :param img_rows: image height
        :param img_cols: image width
        :return: tuple(<np.array>) coordinates for extended line
        """
        new_row_coords = int(0)
        new_col_coords = int(a_coords * new_row_coords + b_coords)
        extended_row_coords = int(img_rows - 1)
        extended_col_coords = int(a_coords * extended_row_coords + b_coords)
        # ! Be careful the below form is in (col, row). It's made in order to pass it to the OpenCV functions
        return (new_col_coords, new_row_coords), (extended_col_coords, extended_row_coords)

