# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import cv2
from bpdb import set_trace
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import chainer
from chainer import training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F


class DNNRegression(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
                l1=L.Linear(None, n_units),
                l2=L.Linear(None, n_units),
                l3=L.Linear(None, n_out),
                )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)



def get_feature_vector(img_path, bins=256):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError
    BGR_splited = cv2.split(img)
    hist_list = list()
    for color_arr in BGR_splited:
        hist = np.histogram(color_arr, bins=bins)[0]
        hist_list.extend(hist / np.linalg.norm(hist))
    return hist_list

def get_feature_vector2(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError
    return np.mean(img)

#def target_val_from_file_name(file_name_):
#    return np.array([float(file_name_.split('__')[0].split('_')[-1])], np.float32)
#

def target_val_from_file_name(file_name_):
    gamma = float(file_name_.split('__')[0].split('_')[-1])
    a = float(os.path.splitext(file_name_.split('__')[1])[0].split('_')[1])
    return np.array([gamma, a], np.float32)

def make_this_training_data(items_dir, feature_vector_func):
    this_training_data_dict_list = list()
    for file_name_ in os.listdir(items_dir):
        try:
            this_training_data_dict_list.append(
                    {
                        file_name_: {
                            'feature_vector': feature_vector_func(os.path.join(items_dir, file_name_)),
                            'target': target_val_from_file_name(file_name_)
                        }
                    }
                )
        except ValueError:
            pass
    return this_training_data_dict_list

def make_whole_training_data(items_dir_list, feature_vector_func):
    training_data_dict = dict()
    for items_dir in items_dir_list:
        dirs_list = filter(lambda f: os.path.isdir(os.path.join(items_dir, f)), os.listdir(items_dir))
        dirs_list = map(lambda d: os.path.join(items_dir, d), dirs_list)
        for dir_ in dirs_list:
            print('Making feature vector of images in {}.'.format(dir_))
            training_data_dict[dir_] = make_this_training_data(dir_, feature_vector_func)
                
    return training_data_dict


def preprocess(items_dir_list, feature_vector_func, training_data_save_dir='storage/data', training_data_file='training_data.pickle'):
    if not os.path.exists(training_data_save_dir):
        os.makedirs(training_data_save_dir)

    if not os.path.exists(os.path.join(training_data_save_dir, training_data_file)):
        training_data_dict = make_whole_training_data(items_dir_list, feature_vector_func)
        try:
            with open(os.path.join(training_data_save_dir, training_data_file), 'wb') as f:
                pickle.dump(training_data_dict, f)
        except:
            print("Warning: training_data_dict hasn't been saved.")
            pass
    else:
        ans = input('training_data.pickle already exists. Overwrite?[y/n]')
        if ans == 'y':
            training_data_dict = make_whole_training_data(items_dir_list, feature_vector_func)
            try:
                with open(os.path.join(training_data_save_dir, training_data_file), 'wb') as f:
                    pickle.dump(training_data_dict, f)
            except:
                print("Warning: training_data_dict hasn't been saved.")
                pass
        else:
            with open(os.path.join(training_data_save_dir, training_data_file), 'rb') as f:
                training_data_dict = pickle.load(f)
    return training_data_dict


def split_data_from_dict(training_data_dict):
    pass

def get_feature_vector_hsv(img_path):
    return np.array([0])


def split_train_test(pickle_path):
    with open(pickle_path, 'rb') as f:
        training_data_dict = pickle.load(f)

    N = len(training_data_dict.keys())
    train_dict = dict()
    test_dict = dict()
    cnt = 0
    for key, val in training_data_dict.items():
        if 'agejo' in key:
            continue
        print(key)
        if cnt >= int(N/2):
            train_dict[key] = list()
            for target_dict in val:
                train_dict[key] = val
        else:
            test_dict[key] = list()
            for target_dict in val:
                test_dict[key] = val
        cnt += 1

    with open('storage/data/train_dict_darker.pickle', 'wb') as f:
        pickle.dump(train_dict, f)

    with open('storage/data/test_dict_darker.pickle', 'wb') as f:
        pickle.dump(test_dict, f)








if __name__ == '__main__':
    items_dir_list = ['storage/image/agejo/', 'storage/image/bijin/']
    training_data_save_dir = 'storage/data/'
    training_data_file = 'training_data.pickle'
    feature_vector_func = get_feature_vector_hsv

    training_data_dict = preprocess(items_dir_list, feature_vector_func)

    split_train_test(os.path.join(training_data_save_dir, training_data_file))
    ###########################################
    ###########################################


    print(training_data_dict.keys())
    set_trace()


    x_data = map(lambda d: d['feature_vector'], training_data_dict.values())
    x_data = list(x_data)
    print(x_data)



    #set_trace()

    #img_dir_path = ['storage/image/agejo/gamma_corrected_12/']
    #vivid_val_dict = dict()
    #feature_vectors = list()
    #for img_dir in img_dir_path:
    #    vivid_val_dict[img_dir] = list()
    #    for i, img_name in enumerate(os.listdir(img_dir)):
    #        try:
    #            feature_vector = get_feature_vector2(os.path.join(img_dir, img_name))
    #            print(feature_vector)
    #            feature_vectors.append(feature_vector)
    #            #if i % 200 == 0:
    #            #    plt.scatter(np.arange(len(feature_vector)), feature_vector)
    #            #    plt.ylim(0.0, 0.2)
    #            #    plt.savefig('1')
    #            #    plt.show()
    #            vivid_val_dict[img_dir].append(feature_vector)
    #        except ValueError:
    #            continue
        #else:
        #        plt.scatter(np.arange(len(feature_vector)), feature_vector)
        #        plt.ylim(0.0, 0.2)
        #        plt.savefig('2')
        #        plt.show()

    #for img_dir in img_dir_path:
    #    pca = PCA(n_components=1)
    #    feature_vectors = vivid_val_dict[img_dir]
    #    pca.fit(feature_vectors)
    #    Xd = pca.fit_transform(feature_vectors)
    #    plt.scatter(np.arange(len(Xd)), Xd)
    #    #plt.scatter(Xd[:, 0], Xd[:, 1])
    #    plt.show()

    for img_dir in img_dir_path:
        feature_vectors = vivid_val_dict[img_dir]
        plt.scatter(np.arange(len(feature_vectors)), feature_vectors)
        plt.show()


    model = DNNRegression()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=1)
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')

    trainer.extend(extentions.Evaluator(test_iter, model, device=10))
    trainer.extend(extentions.dump_graph('main/loss'))
    trainer.extend(
            extensions.PrintReport(
                [
                    'epoch', 'main/loss', 'validation/main/loss', 
                'main/accuracy', 'validation/main/accuracy'
                ]
            )
        )
    trainer.extend(extensions.ProgressBar())
    trainer.run()
