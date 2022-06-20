import numpy as np
import os
import torch
import torch.nn.functional as F

train_path = {f'data_batch_{i}': f'dataset/cifar10/cifar-10-batches-py/data_batch_{i}' for i in range(1,6)}
test_path = {'test_batch' : 'dataset/cifar10/cifar-10-batches-py/test_batch'}
data_dict_keys = {'batch_label':b'batch_label', 'labels':b'labels', 'data':b'data', 'filenames':b'filenames'}

def unpickle(file) -> dict:
    """
    unpickle a file
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_dataset(data_batch_id=None, isTrain=True):
    """
    return a batch's label and data
    """
    if isTrain:
        assert (data_batch_id >= 0 and data_batch_id <= 5)
        data_path = train_path['data_batch_{}'.format(data_batch_id)]
        img_dict = unpickle(data_path)

        labels = img_dict[data_dict_keys['labels']]
        data = img_dict[data_dict_keys['data']]

    else:
        assert (data_batch_id is None)
        data_path = test_path['test_batch']
        img_dict = unpickle(data_path)

        labels = img_dict[data_dict_keys['labels']]
        data = img_dict[data_dict_keys['data']]

    labels = np.array(labels)
    data = np.array(data)
    return labels, data

def get_img(data_batch_id:int, img_idx:int):
    """
    return a image's label and data
    """
    assert (data_batch_id >= 0 and data_batch_id <= 5)
    assert (img_idx >= 0 and img_idx < 10000)

    data_path = train_path['data_batch_{}'.format(data_batch_id)]
    img_dict = unpickle(data_path)

    img_label = img_dict[data_dict_keys['labels']][img_idx]
    img_data = img_dict[data_dict_keys['data']][img_idx]
    img_data = np.asarray(img_data).reshape(3, 32, -1)
    
    return (img_label, img_data)

def get_test_img(img_idx:int):
    """
    return a image's label and data in the testset
    """
    assert (img_idx >= 0 and img_idx < 10000)

    data_path = test_path['test_batch']
    img_dict = unpickle(data_path)

    img_label = img_dict[data_dict_keys['labels']][img_idx]
    img_data = img_dict[data_dict_keys['data']][img_idx]
    img_data = np.asarray(img_data).reshape(3, 32, -1)
    
    return (img_label, img_data)


def img_axisTransform(img_data):
    """
    swap the color channel to 3rd axis
    """
    assert (img_data.shape[0] == 3)
    img = np.swapaxes(img_data, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img

def display_image(img_data):
    """
    show a image with W*H*C
    """
    from PIL import Image
    display = img_axisTransform(img_data)
    display = Image.fromarray(display)
    display.show()
    return display

def split_ssl_data(data, target, num_labels, numClasses, index=None, include_lb_in_ulb=True):
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, = sample_labeled_data(data, target, num_labels, numClasses, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_in_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]

def sample_labeled_data(data, target,
                        num_labels, num_classes,
                        index=None, name=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if index is not None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    save_dir = "./data"
    dump_path = os.path.join(save_dir, str(num_labels)+'sampled_label_idx.npy')

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_data = data[lb_idx]
        lbs = target[lb_idx]
        return lb_data, lbs, lb_idx

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])
    

    np.save(dump_path, np.array(lb_idx))

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def consistency_loss(logits_s, logits_w, class_acc, p_target, p_model, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long(), p_model

    else:
        assert Exception('Not Implemented consistency_loss')



if __name__ == '__main__':
    # filename = 'dataset/cifar10/cifar-10-batches-py/data_batch_1'
    filename = 'dataset/cifar10/cifar-10-batches-py/test_batch'
    pic_dict = unpickle(filename)
    """ dict_keys([b'batch_label', b'labels', b'data', b'filenames']) """
    a = np.array(pic_dict[b'data'])
    print(a.shape)

