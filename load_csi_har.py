import numpy as np
import glob
import os
import csv


class CSIModelConfig:
    """
    class for Human Activity Recognition ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
    Using CSI (Channel State Information)
       I edited the code provided on https://github.com/ludlows.
    Args:
        win_len   :  integer (500 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """

    def __init__(self, win_len=300, step=50, thrshd=0.6):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
        # self._downsample = downsample

    def preprocessing(self, raw_folder, save=False, wrt=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI
            save      : choose if save the numpy array
        """
        if wrt==True:
            train_tuple = extract_csi_train(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
            test_tuple = extract_csi_test(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
            return train_tuple, test_tuple
        else:
            numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
            # if self._downsample > 1:
            #     return tuple([v[:, ::self._downsample,...] if i%2 ==0 else v for i, v in enumerate(numpy_tuple)])
            return numpy_tuple

    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_lie_down.npz', 'x_fall.npz', 'x_bend.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz')
        """
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for lie down, fall, bend, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        # if self._downsample > 1:
        #     x = [arr[:,::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:, i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)


def extract_csi_train(raw_folder, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label_train(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


def extract_csi_test(raw_folder, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label_test(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


def extract_csi_by_label_test(raw_folder, label, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['lie down', 'fall', 'bend', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError(
            "The label {} should be among 'lie down','fall','bend','run','sitdown','standup','walk'".format(labels))

    data_path_pattern = os.path.join(raw_folder, label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    input_csv_files = [s for s in input_csv_files if "user_1" not in s]
    input_csv_files = [s for s in input_csv_files if "user_2" not in s]
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder, label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    annot_csv_files = [s for s in annot_csv_files if "user_1" not in s]
    annot_csv_files = [s for s in annot_csv_files if "user_2" not in s]
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}.npz".format(label), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def extract_csi_by_label_train(raw_folder, label, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['lie down', 'fall', 'bend', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError(
            "The label {} should be among 'lie down','fall','bend','run','sitdown','standup','walk'".format(labels))

    data_path_pattern = os.path.join(raw_folder, label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    input_csv_files = [s for s in input_csv_files if "user_3" not in s]
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder, label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    annot_csv_files = [s for s in annot_csv_files if "user_3" not in s]
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}.npz".format(label), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def extract_csi(raw_folder, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=300, thrshd=0.6, step=50):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['lie down', 'fall', 'bend', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError(
            "The label {} should be among 'lie down','fall','bend','run','sitdown','standup','walk'".format(labels))

    data_path_pattern = os.path.join(raw_folder, label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder, label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}.npz".format(label), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.8, seed=200):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    if train_portion!=1.0:
        np.random.seed(seed=seed)
        x_train = []
        x_valid = []
        y_valid = []
        y_train = []

        for i, x_arr in enumerate(numpy_tuple):
            index = np.random.permutation([i for i in range(x_arr.shape[0])])
            split_len = int(train_portion * x_arr.shape[0])
            x_train.append(x_arr[index[:split_len], ...])
            tmpy = np.zeros((split_len, 7))
            tmpy[:, i] = 1
            y_train.append(tmpy)
            x_valid.append(x_arr[index[split_len:], ...])
            tmpy = np.zeros((x_arr.shape[0] - split_len, 7))
            tmpy[:, i] = 1
            y_valid.append(tmpy)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_valid = np.concatenate(x_valid, axis=0)
        y_valid = np.concatenate(y_valid, axis=0)

        index = np.random.permutation([i for i in range(x_train.shape[0])])
        x_train = x_train[index, ...]
        y_train = y_train[index, ...]
        return x_train, y_train, x_valid, y_valid
    else:
        np.random.seed(seed=seed)
        x_train = []
        y_train = []

        for i, x_arr in enumerate(numpy_tuple):
            index = np.random.permutation([i for i in range(x_arr.shape[0])])
            split_len = int(train_portion * x_arr.shape[0])
            x_train.append(x_arr[index[:split_len], ...])
            tmpy = np.zeros((split_len, 7))
            tmpy[:, i] = 1
            y_train.append(tmpy)
            tmpy = np.zeros((x_arr.shape[0] - split_len, 7))
            tmpy[:, i] = 1

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        index = np.random.permutation([i for i in range(x_train.shape[0])])
        x_train = x_train[index, ...]
        y_train = y_train[index, ...]
        return x_train, y_train


def merge_csi_label(csifile, labelfile, win_len=300, thrshd=0.6, step=50):
    """
    Merge CSV files into a Numperrory Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[0:52]
            csi.append(line_array[np.newaxis, ...])
    csi = np.concatenate(csi, axis=0)
    assert (csi.shape[0] == activity.shape[0])
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index + win_len]
        if np.sum(cur_activity) < thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 52))
        cur_feature[0] = csi[index:index + win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)


if __name__=="__main__":
    cfg = CSIModelConfig(win_len=300, step=50, thrshd=0.6)
    train_tuple, test_tuple = cfg.preprocessing("/mnt/data1/dvarga/Datasets/CSI-HAR-Dataset", save=True, wrt=True)

    x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = train_tuple
    x_train, y_train = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk), train_portion=1.0, seed=200)

    x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = test_tuple
    x_test, y_test = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk), train_portion=1.0, seed=200)

    # fit and evaluate a model
    verbose, epochs, batch_size = 0, 300, 64
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]