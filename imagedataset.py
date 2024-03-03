import pandas as pd
import numpy as np
import os
import cv2


class ImageDataset:
    def __init__(self, folder_path, seq_path):
        self.folder_path = folder_path
        self.image_files = sorted(os.listdir(folder_path))
        self.seq_path = seq_path
        self.current_index = 0

        calib = pd.read_csv(self.seq_path + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3, 4))
        self.camera_matrix = self.P0[:, :3]
        self.lenght = self.image_files.__len__()

    def get_images(self):
        if self.current_index + 1 < len(self.image_files):
            image1 = cv2.imread(os.path.join(self.folder_path, self.image_files[self.current_index]))
            image2 = cv2.imread(os.path.join(self.folder_path, self.image_files[self.current_index + 1]))
            self.current_index += 1
            return image1, image2
        else:
            return None