'''
    Author: 
        Michaelfi
        
    Date: 
        3.7.18
    
    Description: 
        This file contains utils for creating data objects from GoT database JSON files
    
    Python Version:
        3.5
    
'''

import json
import os
import urllib
import tensorflow as tf
import PIL
import numpy as np

PAD = False

GOT_NAME_LIST = ["Aerys Targaryen",
                 "Archmaester Ebrose",
                 "Arya Stark",
                 "Benjen Stark",
                 "Beric",
                 "Bran Stark",
                 "Brienne of Tarth",
                 "Bronn",
                 "Catelyn Stark",
                 "Cersei Lannister",
                 "Daario Naharis",
                 "Daenerys Targaryen",
                 "Ed Sheeran",
                 "Ellaria Sand",
                 "Euron Greyjoy",
                 "Gendry",
                 "Gilly",
                 "Grand Maester Pycelle",
                 "Grey Worm",
                 "Hodor",
                 "Hot Pie",
                 "House frey",
                 "Irri",
                 "Jaime Lannister",
                 "Joer Mormont",
                 "Joffrey Baratheon",
                 "Jojen Reed",
                 "Jon Snow",
                 "Jorah Mormont",
                 "Lady Mormont",
                 "Lysa Arryn",
                 "Meera Reed",
                 "Melisandre",
                 "Missandei",
                 "Myrcella Baratheon",
                 "Ned Stark",
                 "Night king",
                 "Oberyn Martell",
                 "Olenna Tyrell",
                 "Olly",
                 "Peytr Baelish",
                 "Podrick",
                 "Qyburn",
                 "Ramsay Bolton",
                 "Renly Baratheon",
                 "Rhaegar Targaryen",
                 "Rickon Stark",
                 "Robb Stark",
                 "Robert Baratheon",
                 "Roose Bolton",
                 "Samwell Tarly",
                 "Sansa Stark",
                 "Ser Davos",
                 "Stannis Baratheon",
                 "the hound",
                 "the mountain",
                 "Theon Greyjoy",
                 "Thoros",
                 "Tommen Baratheon",
                 "Tormund",
                 "Tycho Nestoris",
                 "Tyene",
                 "Tyrion Lannister",
                 "Tywin Lannister",
                 "Varys",
                 "Walder Frey",
                 "Yara Greyjoy",
                 "Unknown Person",
                 "Unknown - not a face"]

def get_json_data(path):
    '''
    This function will create a dict based on a JSON file with multiple objects
    
    param path:
        A path to a valid JSON file
        
    returns:
        List of items holding the json data
    '''
    data = []
    
    with open(path, 'r') as f:
        for line in f:
            data_t = json.loads(line)
            if data_t:
                data.append(data_t)
    
    return data

def load_images(data, num_of_pics):
    '''
    This function will load a dataset the size of num_of_pics
    
    param data:
        A list of dictionaries of json data including the content path and label.
        i.e.:
            {'content': 'http://www.seas.upenn.edu/~daphnei/images/ep7/ep7.mov.Scene-649-OUT/00.jpg',
            'annotation': {'notes': '', 'label': ['Daenerys Targaryen']}, 'extras': None}
    param num_of_pics:
        Number of pictures to get from the dataset
        
    retunrs:
        list containing file names used for data
    
    '''
    files = []
    if (num_of_pics > len(data)): 
        raise Exception("Number of pictures requested {} is bigger than dataset amount {}".format(num_of_pics, len(data)))
    
    # Download pictures according to size of data requested
    
    if not (os.path.isdir("data")):
        os.mkdir("data")
    
    for entry in range(num_of_pics):
        file_name = ("data/data_%s.jpg" % entry)
        files.append(file_name)
        if os.path.exists(file_name):
            continue
        urllib.request.urlretrieve(data[entry]['content'], file_name)
        
    
    return files

def load_images_and_get_data(data, num_of_pics, num_of_test, flip=False):
    '''
    This function is a wraps load_images and gives a dictionay used for train and validation
    param num_of_pics:
        Number of pictures to be used for training set
    param num_of_test:
        Number of pictures to be used for validation set
    flip:
        multiplies training set by 2 by using data augmentation by flipping the picture.
        
    retunrs:
        dictionary containing training and evaluation data in the following form:
        data_dict['X_train'] = trainig examplers, data_dict['y_train']= training labels
        data_dict['X_eval'] = validation examples, data_dict['y_eval']= validation labels
    '''
    
    data_dict = {}
    num_train = num_of_pics - num_of_test
    num_train_t = num_train
    if (flip):
        num_train *= 2
    data_dict['X_train'] = np.zeros((num_train, 96, 96, 3), dtype='f')
    data_dict['y_train'] = np.zeros((num_train), dtype='f')
    data_dict['X_eval'] = np.zeros((num_of_test, 96, 96, 3), dtype='f')
    data_dict['y_eval'] = np.zeros((num_of_test), dtype='f')
    
    load_images(data, num_of_pics)
    
    for entry in range(num_of_pics):
        file_name = ("data/data_%s.jpg" % entry)
        pic = PIL.Image.open(file_name)
        if entry < num_train_t: 
            data_dict['X_train'][entry] = np.array(pic, dtype='f')
            data_dict['y_train'][entry] = GOT_NAME_LIST.index(data[entry]['annotation']['label'][0])
            if flip:
                data_dict['X_train'][entry + num_train_t] = np.fliplr(np.array(pic, dtype='f'))
                data_dict['y_train'][entry + num_train_t] = GOT_NAME_LIST.index(data[entry]['annotation']['label'][0])
        else:
            data_dict['X_eval'][entry - num_train_t] = np.array(pic, dtype='f')
            data_dict['y_eval'][entry - num_train_t] = GOT_NAME_LIST.index(data[entry]['annotation']['label'][0])
    
    if PAD:
        data_dict['X_train'] = np.pad(data_dict['X_train'], ((0,0), (2,2), (2,2), (0,0)), mode='constant')
        data_dict['X_eval'] = np.pad(data_dict['X_eval'], ((0,0), (2,2), (2,2), (0,0)), mode='constant')
    return data_dict



def load_specific_image(data, num):
    X = np.zeros((1, 96 ,96 ,3), dtype='f')
    file_name = ("data/data_%s.jpg" % num)
    urllib.request.urlretrieve(data[num]['content'], file_name)
    pic = PIL.Image.open(file_name)
    X[0] = np.array(pic, dtype='f')
    
    if PAD:
        X = np.pad(X, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
    
    return X

def num_to_name(num):
    return GOT_NAME_LIST[num]
                     

            
            
    
    