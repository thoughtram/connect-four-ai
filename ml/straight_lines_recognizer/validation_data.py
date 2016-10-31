import numpy as np

def get_validation_train_data():
    return  np.array([
                      [[[1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,1,1,1,1,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,1,1,1,1,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,1],
                        [0,0,0,0,1,0],
                        [0,0,0,1,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,1]]],
                      [[[0,0,0,0,0,0],
                        [1,1,1,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,1,1,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,1,0,0]]],
                      [[[0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,1,0,0],
                        [0,0,1,0,0,0],
                        [0,1,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],

                      [[[1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,1,0,1,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,1,0,1,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,1,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,1,1,0,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,1],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [1,0,0,0,0,0]]],
                      [[[0,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,1],
                        [0,0,0,0,1,1],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [1,1,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,1,1,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,0,1,1],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,1,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,1,1],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [1,0,1,1,0,0],
                        [1,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [1,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,0,0,1],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,1,0,0],
                        [0,1,0,1,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,1,0,0,1],
                        [0,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]]], "float32")

def get_validation_target_data_2d():
    return  np.array([
                      [[[1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,1,1,1,1,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,1,1,1,1,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,1],
                        [0,0,0,0,1,0],
                        [0,0,0,1,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,1,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [1,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,1],
                        [0,0,0,0,0,1]]],
                      [[[0,0,0,0,0,0],
                        [1,1,1,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,1,1,1,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,1,0,0]]],
                      [[[0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,1,0,0],
                        [0,0,1,0,0,0],
                        [0,1,0,0,0,0],
                        [1,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],

                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]],
                      [[[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]]]], "float32")

def get_validation_target_data_1d():
    return np.array([1,1,1,1,1,1,1,1,1,1,
                     1,1,1,1,1,1,1,1,1,1,
                     0,0,0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,0,0], "float32")


def get_validation_data_tuple():
    return (get_validation_train_data(), get_validation_target_data_2d(), get_validation_target_data_1d())
