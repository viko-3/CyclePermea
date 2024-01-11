# -*- coding: utf-8 -*-
import argparse

from transformers import AutoTokenizer

from dataset import SMILESDataset, collate_wrapper
from torch.utils.data import DataLoader
import pandas as pd
from model import Peptide_Regression
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_vocabulary, str2bool, define_optimizer
from train import train
import datetime
from torch.utils.tensorboard import SummaryWriter

import yaml
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def set_log(config):
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('../log/runs/{}'.format(current_time))

    filename = "../log/runs/{}.txt".format(current_time)
    with open(filename, "w") as f:
        f.write("Config:\n")
        f.write(config['data_yaml']['text_data_yaml'])
        f.write(yaml.dump(config))

    return writer, filename


def add_log(filename, config):
    with open(filename, "a") as f:
        f.write("\n")
        f.write(yaml.dump(config))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Peptide Regression')
    parser.add_argument('--all_config', type=str, default='config/model.yaml',
                        help='Path to the config file.')

    parser.add_argument("--text_data_yaml", type=str, default=None),
    parser.add_argument("--text_model_yaml", type=str, default=None),

    parser.add_argument("--use_text_info", type=str2bool, default=True),
    parser.add_argument("--feature_cmb_type", type=str, default='attention'),
    # 可以根据需要添加更多的参数
    parser.add_argument("--CL_margin", type=float, default=1, help="contrastive_learning_margin")
    return parser.parse_args()


def main():
    args = parse_arguments()
    all_config = load_config(args.all_config)

    # 用args覆盖config
    all_config['model_yaml']['use_text_info'] = args.use_text_info
    if args.text_data_yaml is not None:
        all_config['data_yaml']['text_data_yaml'] = args.text_data_yaml

    if args.text_model_yaml is not None:
        all_config['model_yaml']['text_model_yaml'] = args.text_model_yaml

    if args.feature_cmb_type is not None:
        all_config['model_yaml']['feature_cmb_type'] = args.feature_cmb_type

    writer, log_file = set_log(all_config)

    data_yaml_config = all_config['data_yaml']
    model_yaml_config = all_config['model_yaml']

    text_data_yaml = data_yaml_config['text_data_yaml']

    text_model_yaml = model_yaml_config['text_model_yaml']

    model_yaml_config['CL_margin'] = args.CL_margin

    # 读取数据的yaml内容
    text_data_config = load_config(text_data_yaml)
    text_data_type = text_data_config['data_type']

    target_type = text_data_config['target']  # 要预测的回归value的类型

    # 读取模型的yaml内容
    text_model_config = load_config(text_model_yaml)
    text_model_type = text_model_config['model_type']
    batch_size = text_model_config['batch_size']

    # 训练集
    text_train_df = pd.read_csv(text_data_config['train_data_path'])
    train_text_data_list = text_train_df[text_data_type].values
    train_targets = text_train_df[target_type].values
    train_cyclepeptideID = text_train_df['CycPeptMPDB_ID'].values

    # 测试集
    text_test_df = pd.read_csv(text_data_config['test_data_path'])
    test_text_data_list = text_test_df[text_data_type].values
    test_targets = text_test_df[target_type].values
    test_cyclepeptideID = text_test_df['CycPeptMPDB_ID'].values

    # 验证集
    text_val_df = pd.read_csv(text_data_config['val_data_path'])
    val_text_data_list = text_val_df[text_data_type].values
    val_targets = text_val_df[target_type].values
    val_cyclepeptideID = text_val_df['CycPeptMPDB_ID'].values

    add_log(log_file, text_data_config)
    add_log(log_file, model_yaml_config)

    train_HELM_list = text_train_df['HELM'].values
    test_HELM_list = text_test_df['HELM'].values
    val_HELM_list = text_val_df['HELM'].values
    if text_model_type == 'PubChemLM':
        vocab = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    elif text_model_type == 'ChemBERTa':
        vocab = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    else:
        vocab = get_vocabulary(list(train_text_data_list) + list(test_text_data_list) + list(val_text_data_list),
                               text_data_config)
    train_dataset = SMILESDataset(train_text_data_list,
                                  train_HELM_list,
                                  train_targets,
                                  vocab,
                                  train_cyclepeptideID,
                                  text_data_config,
                                  smiles_augment=True)
    test_dataset = SMILESDataset(test_text_data_list,
                                 test_HELM_list,
                                 test_targets,
                                 vocab,
                                 test_cyclepeptideID,
                                 text_data_config,
                                 smiles_augment=True)
    val_dataset = SMILESDataset(val_text_data_list,
                                val_HELM_list,
                                val_targets,
                                vocab,
                                val_cyclepeptideID,
                                text_data_config,
                                smiles_augment=True)

    # 建立dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper(vocab),
                              drop_last=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_wrapper(vocab))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_wrapper(vocab))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Peptide_Regression(all_model_config=model_yaml_config,
                               text_vocab=vocab,
                               text_model_config=text_model_config,
                               text_data_config=text_data_config)
    with open(log_file, "a") as f:
        f.write('Network structure:\n')
        f.write(str(model))
        f.write('\n\n')

    # initialize_model_parameters(model)
    model.to(device)
    MSE_criterion = nn.MSELoss()
    HELM_criterion = nn.CrossEntropyLoss()

    text_optimizer, other_optimizer = define_optimizer(model, text_model_config,
                                                                        use_text=model_yaml_config['use_text_info'],
                                                                        )
    train(model=model, train_loader=train_loader,
          test_loader=test_loader,
          val_loader=val_loader,
          text_optimizer=text_optimizer,
          other_optimizer=other_optimizer,
          regression_criterion=MSE_criterion,
          HELM_criterion=HELM_criterion,
          model_config=model_yaml_config,
          device=device,
          epochs=text_model_config['epochs'],
          writer=writer,
          log_file=log_file)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
