import os
import glob
import csv
import config as cf
from utils import *

class Generator:
    def __init__(self, use):
        self.data_dir = os.path.join(cf.ROOT_DIR, "input")
        self.epoch = cf.EPOCH
        self.batch = cf.BATCH
        self.data_dic = {}

        if use == "train":
            with open(cf.TRAIN_DATA_CSV_PATH) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data_dic[row[0]] = [float(row[1]), float(row[2]), float(row[3])]

        elif use == "val":
            with open(cf.VAL_DATA_CSV_PATH) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data_dic[row[0]] = [float(row[1]), float(row[2]), float(row[3])]
        elif use == "test":
            with open(cf.TEST_DATA_CSV_PATH) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data_dic[row[0]] = [float(row[1]), float(row[2]), float(row[3])]
        else:
            raise

        self.steps = len(self.data_dic) // self.batch

        self.data_lists = list(self.data_dic.keys())

    def __call__(self):
        while True:
            for step in range(self.steps):
                start_idx = step * self.batch
                end_idx = (step + 1) * self.batch

                inputs = []
                targets = []

                for i in range(start_idx, end_idx):
                    # files = glob.glob(os.path.join(self.data_dir, self.data_lists[i], "*"))
                    files = glob.glob(os.path.join(self.data_lists[i], "*"))
                    input = []
                    for idx, file in enumerate(files):
                        if idx == cf.FRAME:
                            break
                        volume = vtk_data_loader(file)
                        if cf.CNN_MODEL == "EfficientNet---":
                            volume = toRGB(volume)
                        else:
                            volume = volume.reshape((volume.shape[0], volume.shape[1], volume.shape[2], 1))
                        volume = volume.astype("float32") / 255
                        input.append(volume)

                    inputs.append(np.array(input))
                    targets.append(np.array(self.data_dic[self.data_lists[i]]))

                inputs = np.array(inputs)
                targets = np.array(targets)

                yield inputs, targets

    def __len__(self):
        return len(self.data_dic)

    def __getitem__(self, idx):
        dir_num = self.data_lists[idx]
        label = np.array(self.data_dic[dir_num])
        data_paths = glob.glob(os.path.join(dir_num, "*"))
        input = []
        for idx, file in enumerate(data_paths):
            if idx == cf.FRAME:
                break
            volume = vtk_data_loader(file)
            if cf.CNN_MODEL == "EfficientNet":
                volume = toRGB(volume)
            else:
                volume = volume.reshape((volume.shape[0], volume.shape[1], volume.shape[2], 1))
            volume = volume.astype("float32") / 255
            input.append(volume)

        input = np.array([input])

        return dir_num, input, label 