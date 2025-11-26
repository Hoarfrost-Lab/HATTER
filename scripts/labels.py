import pandas as pd
import json

def get_labels(dtype='unbalanced', dpath=None):
    if dtype == 'unbalanced':
        PATH = 'data/Swissprot/UnbalancedSwissprot/'
    elif dtype == 'balanced':
        PATH = 'data/Swissprot/BalancedSwissprot/'
    elif dpath is not None:
        df = pd.read_csv(dpath)
        unique_labels = pd.unique(df['EC'])
        return unique_labels
    else:
        raise Exception('Must specify balanced or unbalanced dtype or datapath to create global labels') 


    df1 = pd.read_csv(PATH+'train.csv')
    print('Unique train labels: ', len(pd.unique(df1['EC'])))
    df2 = pd.read_csv(PATH+'valid.csv')
    print('Unique validation labels: ', len(pd.unique(df2['EC'])))
    df3 = pd.read_csv('data/Swissprot/test1.csv')
    print('Unique test1 labels: ', len(pd.unique(df3['EC'])))
    df4 = pd.read_csv('data/Swissprot/test2.csv')
    print('Unique test2 labels: ', len(pd.unique(df4['EC'])))

    df = pd.concat([df1, df2, df3, df4])
    
    #for determining good sequence length
    #lengths = []
    #for seq in df['Sequence']:
    #    lengths.append(len(seq))
    #print(pd.Series(lengths).describe())

    unique_labels = pd.unique(df['EC'])
    print('Total unique labels: ', len(unique_labels))

    print('Overlap valid with train: ', len(pd.Index(pd.unique(df1['EC'])).intersection(pd.Index(pd.unique(df2['EC'])))))
    print('Overlap test1 with train: ', len(pd.Index(pd.unique(df1['EC'])).intersection(pd.Index(pd.unique(df3['EC'])))))
    print('Overlap test2 with train: ', len(pd.Index(pd.unique(df1['EC'])).intersection(pd.Index(pd.unique(df4['EC'])))))

    return unique_labels


def convert_string_to_list(string):
    return list(map(int, string.replace('n', '1000').split('.')))

