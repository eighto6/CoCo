import numpy as np
def IA_processing(data,normal_class:int,args):

    normal_idx=np.where(data.labels==0)[0]
    if args.dataset in ['inj_cora','inj_amazon']:
        if normal_class == 0:
            abnormal_idx = np.where(data.labels != normal_class)[0]
        else:
            abnormal_idx = np.where(data.labels == normal_class)[0]
    else:
        abnormal_idx=np.where(data.labels!=0)[0]
    # abnormal_idx=np.where(data.labels!=normal_class)[0]
    data.labels[normal_idx]=0
    data.labels[abnormal_idx]=1


    train_mask = np.zeros(data.labels.shape, dtype='bool')
    val_mask = np.zeros(data.labels.shape, dtype='bool')
    test_mask = np.zeros(data.labels.shape, dtype='bool')
    train_mask[normal_idx] = 1
    train_mask[abnormal_idx] = 1
    val_mask[normal_idx] = 1
    val_mask[abnormal_idx] = 1

    test_mask[normal_idx] = 1
    test_mask[abnormal_idx] = 1

    return data.labels, train_mask, val_mask, test_mask

