import os

if __name__ == "__main__":
    dataPath = './dataset/'
    train = dataPath+'train/'
    val = dataPath + 'val/'

    types = {'train':{'3pn':0, 'nf':0, '1pn':0, 'gv':0, '2pn':0},
             'val':{'3pn':0, 'nf':0, '1pn':0, 'gv':0, '2pn':0}}
    for i,type in enumerate([train, val]):
        for subtype in os.listdir(type):
            newPath = type + subtype
            if i == 0:
                types['train'][subtype] += len(os.listdir(newPath))
            elif i == 1:
                types['val'][subtype] += len(os.listdir(newPath))

    print(types)
