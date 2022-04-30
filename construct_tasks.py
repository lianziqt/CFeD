from PIL import Image
import numpy
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def my_pickle(thing, path):
    with open(path, 'wb') as f:
        pickle.dump(thing, f)

def split_cifar_into334(cifar, type="train"):
    cifar012 = []
    cifar345 = []
    cifar6789 = []

    to_child_dataset = {
        0:cifar012,
        1:cifar012,
        2:cifar012,
        3:cifar345,
        4:cifar345,
        5:cifar345,
        6:cifar6789,
        7:cifar6789,
        8:cifar6789,
        9:cifar6789,
    }

    for i, data in enumerate(cifar['data']):
        label = cifar['labels'][i]
        if len(to_child_dataset[label]) > 100:
            continue
        to_child_dataset[label].append((data, label))
    
    with open('CIFAR10-1-{}'.format(type), 'wb') as f:
        pickle.dump(cifar012, f)
    with open('CIFAR10-2-{}'.format(type), 'wb') as f:
        pickle.dump(cifar345, f)
    with open('CIFAR10-3-{}'.format(type), 'wb') as f:
        pickle.dump(cifar6789, f)


def read_cifar(root):
    cifar = {
        'data':[],
        'labels':[],
    }
    for i in range(1, 6):
        with open(root + '\data_batch_{}'.format(i), 'rb') as f:
            temp = pickle.load(f, encoding='bytes')
            cifar['data'].extend(temp[b'data'])
            cifar['labels'].extend(temp[b'labels'])
    split_cifar_into334(cifar, 'train')
    with open(root + '/test_batch', 'rb') as f:
        temp = pickle.load(f, encoding='bytes')
        cifar['data'].extend(temp[b'data'])
        cifar['labels'].extend(temp[b'labels'])
    split_cifar_into334(cifar, 'eval')
    return cifar

def parse_cifar100(train_path, test_path):
    # train_path = 'cifar-100-python/train'
    train_set = unpickle(train_path)
    # test_path = 'cifar-100-python/test'
    test_set = unpickle(test_path)
    
    print(len(train_set[b'data']))
    print(len(train_set[b'fine_labels']))
    label2data = dict()
    
    for i, data in enumerate(train_set[b'data']):
        label = train_set[b'fine_labels'][i]
        if label not in label2data:
            label2data[label] = []
        label2data[label].append(data)
    print(len(label2data))
    my_pickle(label2data, os.path.join([train_path, "label2data_train"]))
    
    label2data = dict()
    for i, data in enumerate(test_set[b'data']):
        label = test_set[b'fine_labels'][i]
        if label not in label2data:
            label2data[label] = []
        label2data[label].append(data)
    print(len(label2data))
    my_pickle(label2data, os.path.join([train_path, "label2data_test"]))

def parse_caltech256(root):
    img_datas = []
    for (temp, dirs, files) in os.walk(root):
        for img_class in dirs:
            for (temp, temp_dirs, img_files) in os.walk(os.path.join(root, img_class)):
                for img_file in img_files:
                    if not img_file.endswith('.jpg'):
                        print(img_file)
                        continue
                    image= Image.open(os.path.join(root, img_class, img_file))
                    image=image.resize((32,32))
                    x_data = numpy.array(image)
                    x_data= x_data.astype(numpy.float32)
                    if len(x_data.shape)!=3:
                        temp=numpy.zeros((x_data.shape[0],x_data.shape[1],3))
                        temp[:,:,0] = x_data
                        temp[:,:,1] = x_data
                        temp[:,:,2] = x_data
                        x_data= temp
                        x_data=numpy.transpose(x_data,(2,0,1))
                    img_datas.append((x_data, -1))
    with open('caltech256', 'wb') as f:
        pickle.dump(img_datas, f) 

# if __name__ == '__main__':
    # parse_cifar100('cifar-100-python/train', 'cifar-100-python/test')
    # parse_caltech256('256_ObjectCategories')