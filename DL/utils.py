import argparse
import random
import re
from multiprocessing import Pool
from collections import UserList, defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import ast
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from tdc import Evaluator
from transformers import RobertaTokenizerFast
import torch.nn.functional as F

mse_evaluator = Evaluator(name='MSE')
R2_evaluator = Evaluator(name='R2')
pcc_evaluator = Evaluator(name='PCC')
scc_evaluator = Evaluator(name='Spearman')
rmse_evaluator = Evaluator(name='RMSE')
mae_evaluator = Evaluator(name='MAE')


def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return torch.tensor(np.array(fingerprint))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    cls = '<cls>'
    mask = '<mask>'


def smiles_tokenizer(smile):
    "Tokenizes SMILES string"
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens


def hugf_tokenizer(tokenizer, smi):
    tokens = tokenizer(smi, return_tensors='pt', padding=True, max_length=250, truncation=True)
    return tokens['input_ids'][0]


HELM_dict = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    '10': 9,
    '11': 10,
    '12': 11,
    '13': 12,
    '14': 13,
    'R1': 14,
    'R2': 15,
    'R3': 16
}


def map_helm_to_values(helm_string):
    connection_pattern = re.compile(r'\d+:\w+-\d+:\w+')
    helm_string = connection_pattern.findall(helm_string)[0]
    # 移除冒号和横杠，并分割字符串
    elements = helm_string.replace(":", " ").replace("-", " ").split()
    # 映射到字典的值
    return [HELM_dict[element] for element in elements if element in HELM_dict]


def get_tokens(datas, config):
    all_tokens = set()
    for data in datas:
        if config['data_type'] == "SMILES":
            token = set(smiles_tokenizer(data))
        else:
            print("data type error")
        all_tokens.update(token)
    return all_tokens


def load_token(path):
    with open(path, 'rb') as f:
        loaded_vocab = pickle.load(f)
    return loaded_vocab


def save_token(path, vocab):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


class WordVocab:
    @classmethod
    def from_data(cls, data, config, *args, **kwargs):
        chars = get_tokens(data, config)
        if config['load_vocab']:
            old_chars = load_token(config['vocab_path'])
            chars = chars | old_chars
            save_token(config['vocab_path'], chars)
        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk, ss.cls, ss.mask]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    @property
    def cls(self):
        return self.c2i[self.ss.cls]

    @property
    def mask(self):
        return self.c2i[self.ss.mask]


def get_vocabulary(data, config):
    return WordVocab.from_data(data, config)


def extent_mask(pre_text_mask, image_feature):
    batch, text_len = pre_text_mask.size()
    _, image_len, _ = image_feature.size()
    image_mask = torch.ones(batch, image_len).cuda()

    new_mask = torch.cat((image_mask, pre_text_mask), dim=1)

    return new_mask


def kfold_split(k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return kf


def initialize_model_parameters(model, skip_names=['image_model']):
    """
    初始化模型中的参数，可以选择跳过指定名称的参数
    Args:
        model: 要初始化的模型
        init_func: 用于初始化参数的初始化函数
        skip_names: 跳过初始化的参数名称列表
    """

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'ln' in name or 'image_model' in name:
                continue
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # for name, module in model.named_children():
    #     print('1',name)
    #     if name not in skip_names:
    #         for param_name, param in module.named_parameters():
    #             if param.requires_grad and 'ln' not in param_name :
    #                 if 'weight' in param_name:
    #                     print('2',param_name)
    #                     nn.init.xavier_normal_(param)
    #                 elif 'bias' in param_name:
    #                     nn.init.zeros_(param)


import matplotlib.pyplot as plt


def plot_true_vs_predicted(y_true, y_pred):
    """
    绘制真实值与预测值的散点图。

    参数:
    y_true (list): 真实值列表。
    y_pred (list): 预测值列表。
    """
    plt.scatter(y_true, y_pred)  # 绘制散点图
    plt.plot(y_true, y_true, color='red')  # 绘制对角线

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.savefig('res.png')


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return R2_evaluator(y_true, y_pred)


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #plot_true_vs_predicted(y_true, y_pred)
    return mse_evaluator(y_true, y_pred)


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return rmse_evaluator(y_true, y_pred)


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return mae_evaluator(y_true, y_pred)


def pearson_correlation_coefficient(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return pcc_evaluator(y_true, y_pred)


def spearman_correlation_coefficient(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return scc_evaluator(y_true, y_pred)


def define_optimizer(model, text_model_config, use_text=True, use_image=True):
    text_optimizer = optim.AdamW(model.text_model.parameters(), lr=text_model_config['lr']) if use_text else None

    other_modules = [module for name, module in model.named_children() if name not in ['text_model', 'image_model']]
    other_params = [p for module in other_modules for p in module.parameters()]
    other_optimizer = optim.AdamW(other_params, lr=0.0001) if len(other_params) > 0 else None
    return text_optimizer, other_optimizer


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
        else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, weight=None, size_average=True):
        distances = ((output2 - output1)).pow(2).mean(1)  # squared distances

        losses = 0.5 * \
                 (target.float() * distances +
                  (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        losses = weight * losses if weight is not None else losses
        return losses.mean() if size_average else losses.sum()


def tanimoto_similarity(fp1, fp2):
    intersection = sum(1 for a, b in zip(fp1, fp2) if a == b == 1)
    union = sum(1 for a, b in zip(fp1, fp2) if a == 1 or b == 1)
    return intersection / union if union != 0 else 0


def get_contrastiveLoss(feature, aug_feature, weight, margin):
    weight = weight.cuda()
    fg_target = torch.ones(feature.size(0), 1).cuda()
    aug_fg_target = torch.zeros(feature.size(0) - 1, 1).cuda()

    loss1 = ContrastiveLoss(margin=margin)(feature, aug_feature, target=fg_target)
    loss2 = ContrastiveLoss(margin=margin)(feature[1:], aug_feature[:-1], weight=None, target=aug_fg_target)
    CL_loss = (loss1 + loss2) / 2
    return CL_loss


def HELM_acc(predicts, labels):
    predicts = np.array(predicts)
    labels = np.array(labels)
    accuracies = []

    correct = (predicts == labels)
    correct = correct.reshape(4, -1)
    correct_num = np.all(correct, axis=0)
    all_acc = correct_num.sum() / correct.shape[1]
    accuracies.append(all_acc)

    predicts = predicts.reshape(4, -1)
    labels = labels.reshape(4, -1)

    for i in range(4):
        # 比较预测和真实标签
        correct = (predicts[i] == labels[i]).sum().item()

        # 计算准确率
        acc = correct / len(predicts[i])
        accuracies.append(acc)
    return accuracies


def draw_feature_distance(all_feature, distance, similarity):
    import seaborn as sns
    import pandas as pd
    df =pd.DataFrame()
    df['feature_cos_similarity']=distance.numpy()
    df['tarnimoto_similarity'] = similarity.numpy()
    # 保存 DataFrame 到 CSV 文件
    plot_true_vs_predicted(distance, similarity)
    df.to_csv("all.csv", index=False)

    # df = pd.DataFrame(all_feature.numpy())
    # df.to_csv("0_feature.csv", index=False)
    #print(pearson_correlation_coefficient(distance,similarity))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(distance, shade=True, color='blue', label='Feature Similarity')
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.kdeplot(similarity, shade=True, color='orange', label='Tarnimoto Similarity')
    plt.legend()
    plt.savefig('distance.png')


def feature_similarity(feature1, feature2):
    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(feature1, feature2, dim=1)
    return cosine_similarity
