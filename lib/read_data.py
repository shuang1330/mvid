from sklearn.model_selection import train_test_split
import collections
import numpy as np

Datasets = collections.namedtuple('Datasets',['train','test'])

class dataset(object):

    def __init__(self,values,labels,seed=None):
        '''
        input values and labels are numpy arrarys,
        values with shape [num_examples, num_features]
        labels with shape [num_examples, 1]
        '''
        assert values.shape[0] == labels.shape[0]
        self._num_examples = values.shape[0]

        self._values = values
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def values(self):
        return self._values

    @property
    def labels(self):
        return self._values

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size,shuffle=True):
        '''
        return the next 'batch_size' examples from this data set.
        '''
        start = self._index_in_epoch
        #shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._values = self.values[perm0]
            self._labels = self.labels[perm0]
        if start + batch_size > self._num_examples:
            # finish epoch
            self._epochs_completed += 1
            # get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            value_rest_part = self._values[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]
            # shuffle the dataset
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._values = self.values[perm]
                self._labels = self.labels[perm]
            # start the new epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            value_new_part = self._values[start:end]
            label_new_part = self._labels[start:end]
            return np.concatenate((value_rest_part,value_new_part),axis=0),\
            np.concatenate((label_rest_part,label_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._values[start:end], self._labels[start:end]

def read_data_set(data_table,test_size=0.25):
    '''
    convert a pandas dataframe data table into Datasets(dataset,dataset)
    '''
    train, test = train_test_split(data_table,test_size=0.25)
    train_x = np.array(train[[col for col in train.columns if col not in ['INFO']]])
    test_x = np.array(test[[col for col in train.columns if col not in ['INFO']]])
    # for col in train.columns:
    #     if col not in ['INFO']:
    #         train_x[col] = train[col]
    #         test_x[col] = test[col]
    train_y = np.array(train['INFO'])
    test_y = np.array(test['INFO'])
    return Datasets(train=dataset(train_x,train_y),test=dataset(test_x,test_y))
