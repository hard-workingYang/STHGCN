import numpy as np
import math
class StandardScaler():
    def __init__(self, mean, std, fill_zeroes=True):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def to_slice_with_time_embed(data, len, weekLen, dayLen):
    shape = data.shape
    reshapedData = data.reshape(shape[0],shape[1]*shape[2],shape[3])
    sumLen = shape[0]

    dayEmbed = np.zeros((sumLen, dayLen))
    index = 0
    for i in range(sumLen):
        dayEmbed[i][index] = 1
        index = index + 1
        if index >= dayLen:
            index = 0
    
    weekEmbed = np.zeros((sumLen, weekLen))
    index = 0
    for i in range(sumLen):
        weekEmbed[i][index] = 1
        index = index + 1
        if index >= weekLen:
            index = 0

    start = 0
    xData = []
    yData = []
    dayEmbedData = []
    weekEmbedData = []
    while start + len < shape[0]:
        xData.append(reshapedData[start:start + len,:,:])
        dayEmbedData.append(dayEmbed[start:start + len,:])
        weekEmbedData.append(weekEmbed[start:start + len,:])
        yData.append(reshapedData[start + len,None,:,:])
        start = start + 1
    xData = np.array(xData)
    dayEmbedData = np.array(dayEmbedData)
    weekEmbedData = np.array(weekEmbedData)
    yData = np.array(yData)

    xData = [xData, weekEmbedData, dayEmbedData]

    return (xData, yData)

def split_by_percent(data, train_weight, val_weight, test_weight):
    data_len = data.shape[0]
    train_len = (int)(data_len * (train_weight / ((train_weight + val_weight + test_weight) * 1.0)))
    val_len = (int)(data_len * (val_weight / ((train_weight + val_weight + test_weight) * 1.0)))

    train_data = data[:train_len,:]
    val_data = data[train_len:train_len+val_len,:]
    test_data = data[train_len+val_len:,:]
    return train_data, val_data, test_data

def split_list_by_percent(data, train_weight, val_weight, test_weight):
    train_data = []
    val_data = []
    test_data = []
    for d in data:
        tmp_train_data, tmp_val_data, tmp_test_data = split_by_percent(d, train_weight, val_weight, test_weight)
        train_data.append(tmp_train_data)
        val_data.append(tmp_val_data)
        test_data.append(tmp_test_data)
    return train_data, val_data, test_data

class DataLoaderForMergeList(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
        self.size = len(xs[0])
        self.num_batch = math.ceil(self.size / self.batch_size)

        self.xs = xs
        self.ys = ys

    def shuffle(self):

        permutation = np.random.permutation(self.size)

        xs = []
        for x in self.xs:
            xs.append(x[permutation])
        ys = self.ys[permutation]

        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))

                x_i = []
                for x in self.xs:
                    x_i.append(x[start_ind: end_ind, ...])
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


def dataset_partition(data_path, train_weight, valid_weight, test_weight, close_len, weekLen, dayLen, batch_size, valid_batch_size= None, test_batch_size=None, fill_zeroes=None):

    data = np.load(data_path)
    volume_dataX, volume_dataY = to_slice_with_time_embed(data, close_len, weekLen = weekLen, dayLen = dayLen)

    my_volume_train_dataX, my_volume_valid_dataX, my_volume_test_dataX = split_list_by_percent(volume_dataX, train_weight, valid_weight, test_weight)
    my_volume_train_dataY, my_volume_valid_dataY, my_volume_test_dataY = split_by_percent(volume_dataY, train_weight, valid_weight, test_weight)

    data = {}

    data['x_train'] = my_volume_train_dataX
    data['y_train'] = my_volume_train_dataY

    data['x_val'] = my_volume_valid_dataX
    data['y_val'] = my_volume_valid_dataY
    
    data['x_test'] = my_volume_test_dataX
    data['y_test'] = my_volume_test_dataY

    BothFlowScaler = StandardScaler(mean=data['x_train'][0].mean(), std=data['x_train'][0].std(), fill_zeroes=fill_zeroes)

    for category in ['train', 'val', 'test']:
        data['x_' + category][0] = BothFlowScaler.transform(data['x_' + category][0])

    data['train_loader'] = DataLoaderForMergeList(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderForMergeList(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderForMergeList(data['x_test'], data['y_test'], test_batch_size)

    data['Scaler'] = BothFlowScaler
    return data