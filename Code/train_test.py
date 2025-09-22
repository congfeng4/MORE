import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from param import parameter_parser
from utils import (Eu_dis, hyperedge_concat, generate_G_from_H, construct_H_with_KNN, one_hot_tensor, cal_sample_weight)
from utils import save_model_dict

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_test_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx):
    H_tr = []
    H_te = []
    for i in range(len(data_tr_list)):
        H_1 = construct_H_with_KNN(data_tr_list[i], K_neigs=3, split_diff_scale=False, is_probH=True, m_prob=1)

        H_tr.append(H_1)

        H_2 = construct_H_with_KNN(data_te_list[i], K_neigs=3, split_diff_scale=False, is_probH=True, m_prob=1)
        H_te.append(H_2)

    H_train = hyperedge_concat(H_tr[0], H_tr[1], H_tr[2])

    H_test = hyperedge_concat(H_te[0], H_te[1], H_te[2])

    adj_train_list = generate_G_from_H(H_train, variable_weight=False)

    adj_test_list = generate_G_from_H(H_test, variable_weight=False)

    return adj_train_list, adj_test_list


def train_epoch(num_cls, data_list, adj_list, label, one_hot_label,
                sample_weight, model_dict, optim_dict, train_MOSA=True):
    loss_dict = {}

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()

    if train_MOSA and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["E{:}".format(i + 1)](data_list[i], adj_list))

        new_data = torch.cat([ci_list[0], ci_list[1], ci_list[2]], dim=1)

        c = model_dict["C"](new_data)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(num_cls, data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["E{:}".format(i + 1)](data_list[i], adj_list))

    if num_view >= 2:

        new_data = torch.cat([ci_list[0], ci_list[1], ci_list[2]], dim=1)
        c = model_dict["C"](new_data)
    else:
        c = ci_list[0]

    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c,
               num_epoch_pretrain, num_epoch):
    test_inverval = 50

    model_folder = os.path.join(data_folder, 'models')
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    dim_he_list = [100, 10]
    data_tr_list, data_te_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)

    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx)
    dim_list = [x.shape[1] for x in data_tr_list]
    input_data_dim = [dim_he_list[-1], dim_he_list[-1], dim_he_list[-1]]
    args = parameter_parser()
    model_dict = init_model_dict(input_data_dim, args, num_view, num_class, dim_list, dim_he_list, dim_hvcdn)

    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    print("\nPretrain MOHE...")

    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(num_class, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_MOSA=False)

    print("\nTraining...")

    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch + 1):
        train_epoch(num_class, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_MOSA=True)

        if epoch % test_inverval == 0:
            te_prob = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict)

            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    folder = os.path.join(model_folder, str(1))
    save_model_dict(folder, model_dict)
