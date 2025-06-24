# dataset.py
import torch
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np

def prepare_dataset(args, samples_type='ratio'):
  
    if args.dataset == 'Indian':
        data = loadmat('./data/indian_pines_TAP.mat')
        TR = data['TR']
        TE = data['TE']
        input_data = data['input']  
    elif args.dataset == 'Salinas':
        data = loadmat('./data/salinasTAP15PC.mat')
        TR = data['TR']
        TE = data['TE']
        input_data = data['input']  # Adjust expected shape accordingly
    elif args.dataset == 'Pavia':
        data = loadmat('./data/Pavia_30.mat')
        TR = data['TR']
        TE = data['TE']
        input_data = data['input']  # Adjust expected shape accordingly
    else:
        raise ValueError("Unsupported dataset. Choose from ['Indian', 'Salinas', 'Pavia'].")

    label = TR + TE
    num_classes = np.max(TR)
    
    # Train data change to the ratio of train samples
    if samples_type == 'ratio':
        training_ratio = 1  # Use all available training samples
        print('Train data change to the ratio of train samples: {}'.format(training_ratio))
        train_idx, TR = split_train_data_clssnum(TR, num_classes, training_ratio)

    # Normalize data by band norm
    input_normalize = np.zeros(input_data.shape)
    for i in range(input_data.shape[2]):
        input_max = np.max(input_data[:, :, i])
        input_min = np.min(input_data[:, :, i])
        input_normalize[:, :, i] = (input_data[:, :, i] - input_min) / (input_max - input_min)
    # Data size
    height, width, band = input_data.shape
    print("Dataset: {}".format(args.dataset))
    print("Height={}, Width={}, Bands={}".format(height, width, band))
    # -------------------------------------------------------------------------------
    # Obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train_band, x_test_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, patch=args.patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
    # -------------------------------------------------------------------------------
    # Load data
    x_train = torch.from_numpy(x_train_band.transpose(0, 3, 2, 1)).type(torch.FloatTensor)  # [num_train, bands, patch_size, patch_size]
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [num_train]
    Label_train = Data.TensorDataset(x_train, y_train)
    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)

    x_test = torch.from_numpy(x_test_band.transpose(0, 3, 2, 1)).type(torch.FloatTensor)  # [num_test, bands, patch_size, patch_size]
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [num_test]
    Label_test = Data.TensorDataset(x_test, y_test)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=False)
    return label_train_loader, label_test_loader, band, height, width, num_classes, label, total_pos_true

# Split dataset by training set ratio
def split_train_data_clssnum(gt, num_classes, train_num_ratio):
    train_idx = []

    TR = np.zeros_like(gt)
    for i in range(num_classes):
        idx = np.argwhere(gt == i + 1)
        samplesCount = len(idx)
        # print("Class", i, ":", samplesCount)
        sample_num = np.ceil(train_num_ratio * samplesCount).astype('int32')
        train_idx.append(idx[: sample_num])

        for j in range(sample_num):
            TR[idx[j, 0], idx[j, 1]] = i + 1

    train_idx = np.concatenate(train_idx, axis=0)
    return train_idx, TR

# Locate training and testing samples
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.concatenate((total_pos_train, pos_train[i]), axis=0)  # (num_train,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.concatenate((total_pos_test, pos_test[i]), axis=0)  # (num_test,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    for i in range(num_classes + 1):
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.concatenate((total_pos_true, pos_true[i]), axis=0)
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

# Mirror padding
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # Center region
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # Left mirror
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # Right mirror
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # Top mirror
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # Bottom mirror
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi

# Get patch image data
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image

# Collect training and testing data
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, true_point=None):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)

    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    if true_point is not None:
        x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
        for k in range(true_point.shape[0]):
            x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
        print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
        print("**************************************************")
        return x_train, x_test, x_true
    else:
        print("**************************************************")
        return x_train, x_test

# Labels y_train, y_test
def train_and_test_label(number_train, number_test, num_classes, number_true=None):
    y_train = []
    y_test = []
    for i in range(num_classes):
        y_train.extend([i] * number_train[i])
        y_test.extend([i] * number_test[i])
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))

    if number_true is not None:
        y_true = []
        for i in range(num_classes + 1):
            y_true.extend([i] * number_true[i])
        y_true = np.array(y_true)
        print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
        print("**************************************************")
        return y_train, y_test, y_true
    else:
        print("**************************************************")
        return y_train, y_test