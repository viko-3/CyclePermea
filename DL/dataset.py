import random

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from utils import smiles_tokenizer, hugf_tokenizer, get_fingerprint, map_helm_to_values, tanimoto_similarity

import os

import cv2

from rdkit import Chem


##############################################################################################################
# downside for Text
##############################################################################################################

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, HELM, targets, vocab, cyclepeptideID, text_data_config, smiles_augment = False):
        self.smiles_list = smiles_list
        self.targets = targets
        self.vocab = vocab
        self.max_len = text_data_config['max_len']
        self.augment = text_data_config['augmentation'] and smiles_augment
        self.augment_ratio = text_data_config['augmentation_ratio']
        self.use_HELM = text_data_config['use_HELM']
        self.HELM = HELM
        self.cyclepeptideID = cyclepeptideID
    def __len__(self):
        return len(self.smiles_list)

    def randomize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        smi = Chem.MolToSmiles(mol, doRandom=True)
        if Chem.MolFromSmiles(smi):
            return smi
        else:
            return smiles

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        # add fingerprint
        fingerprint = get_fingerprint(smi)

        if self.augment:
            if random.random() > (1 - self.augment_ratio):
                augment_smi = self.randomize_smiles(smi)
                augment_fingerprint = get_fingerprint(augment_smi)
        else:
            augment_smi = None
            augment_fingerprint = None

        if isinstance(self.vocab, (PreTrainedTokenizer, PreTrainedTokenizerFast)):  # 用huggingface的
            smi_token_ids = hugf_tokenizer(self.vocab, smi)
            augment_smi_token_ids = hugf_tokenizer(self.vocab, augment_smi) if augment_smi is not None else None

        else:  # 用自己的分词器
            smi = smiles_tokenizer(smi)
            smi_token_ids = [self.vocab.c2i[char] for char in smi]
            # data_token_ids.append(self.vocab.eos)  # 在末尾添加
            smi_token_ids.insert(0, self.vocab.bos)  # 在开头添加

        actual_length = len(smi_token_ids)  # 记录实际长度

        target = self.targets[idx]
        data_token_ids = torch.tensor(smi_token_ids, dtype=torch.long)
        augment_data_token_ids = torch.tensor(augment_smi_token_ids,
                                              dtype=torch.long) if augment_smi_token_ids is not None else None
        target = torch.tensor(target, dtype=torch.float32)


        if self.use_HELM:
            HELM = self.HELM[idx]
            HELM = map_helm_to_values(HELM)
            HELM = torch.tensor(HELM, dtype=torch.long)
        else:
            HELM = None

        return data_token_ids, augment_data_token_ids, target, actual_length, fingerprint, augment_fingerprint, HELM


def collate_wrapper(vocab):
    def collate_fn(batch):
        # Sort batch by sequence length
        batch = sorted(batch, key=lambda x: x[3], reverse=True)
        sequences, aug_sequences, targets, lengths, fingerprint, aug_fingerprint, HELM = zip(*batch)

        # Pad sequences
        if isinstance(vocab, (PreTrainedTokenizer, PreTrainedTokenizerFast)):  # 用huggingface的
            seqs_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=vocab.pad_token_id)
            aug_seqs_padded = torch.nn.utils.rnn.pad_sequence(aug_sequences, batch_first=True,
                                                              padding_value=vocab.pad_token_id) if aug_sequences[
                                                                                                       0] is not None else None
        else:
            seqs_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=vocab.pad)

        # Convert lengths to tensor
        lengths = torch.tensor(lengths, dtype=torch.long)

        HELM = torch.stack(HELM) if HELM[0] is not None else HELM

        sim_for_weight = 1-torch.tensor(
            [tanimoto_similarity(fg1, fg2) for fg1, fg2 in zip(fingerprint[1:], aug_fingerprint[:-1])])

        return seqs_padded, aug_seqs_padded, torch.stack(targets), lengths,torch.stack(
            fingerprint), torch.stack(aug_fingerprint), HELM, sim_for_weight

    return collate_fn
