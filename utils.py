from termcolor import colored
import tensorflow as tf
import numpy as np
import csv
import os
import errno
import shutil


def col(x, color):
    """ helper function for color """
    if color == "g":
        real_color = "green"
    elif color == "b":
        real_color = "blue"
    elif color == "r":
        real_color = "red"
    elif color == "y":
        real_color = "yellow"
    else:
        raise ValueError(
            "Color input should be 'g' (green), 'b' (blue), 'r' (red) or 'y' (yellow)")

    return colored(x, real_color)


def parameters_profiler():
    """ helper function for parameters """
    sumz = 0
    print(col("list of parameters:", "b"))
    for i in tf.global_variables():
        print(col(i.name, "g") + col(i.shape, "g") + col(np.prod(
            np.array(i.get_shape().as_list())), "g"))
        sumz += np.prod(np.array(i.get_shape().as_list()))
    print(col("number of parameters: ", "b") + col(sumz, "g"))


def log(dictionary, save_path):
    """ helper function for log """

    filepath = os.path.join(save_path, 'log.csv')
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(filepath, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def file_manager(save_path):
    """ helper function to manage working directories """
    if os.path.exists(save_path):
        print(colored(
            "Directory exists. Enter a string in [Y, yes, y] to override it.", "red"))
        inp = raw_input("Enter key here: ")
        if inp in ["Y", "yes", "y"]:
            print(colored("OK: overriding...", "red"))
            shutil.rmtree(save_path)
        else:
            print(colored("Invalid key: exiting...", "blue"))
            exit()


def generate_points_for_visualization(num_param, num_points, type_vis="linear", range_gen=10):
    """ 
    helper function that generates the plot of the energy landscape of the model. 
    type_vis can be `linear` or `contour` 
    """
    points_collect = []
    coordinates = []
    if type_vis == "linear":
        point_a = np.random.uniform(-range_gen,
                                    range_gen, size=num_param)
        point_b = np.random.uniform(-range_gen,
                                    range_gen, size=num_param)
        for i in range(num_points):
            alpha = i / float(num_points)
            coordinates.append(alpha)
            points_collect.append(
                (1 - alpha) * point_a + alpha * point_b)
    elif type_vis == "contour":
        point_ref = np.random.uniform(-range_gen,
                                      range_gen, size=num_param)
        point_delta = np.random.uniform(-range_gen,
                                        range_gen, size=num_param)
        point_nu = np.random.uniform(-range_gen,
                                     range_gen, size=num_param)
        for i in range(num_points):
            for j in range(num_points):
                alpha = i / float(num_points)
                beta = j / float(num_points)
                points_collect.append(
                    point_ref + alpha * point_delta + beta * point_nu)
        for i in range(num_points):
            alpha = i / float(num_points)
            coordinates.append(alpha)
    points_collect = np.stack(points_collect, axis=0)
    return np.array(coordinates), points_collect
