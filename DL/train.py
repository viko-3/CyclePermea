import os

from model import BiRNNRegression, Transformer, TextCNN
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_vocabulary, r2_score, mse, get_contrastiveLoss
import numpy as np
from tqdm import tqdm
import yaml
from contextlib import nullcontext


def train(model, train_loader, test_loader, val_loader, text_optimizer,other_optimizer,
          regression_criterion,
          HELM_criterion,
          model_config,
          device, epochs, writer, log_file):
    best_test_loss = 1000
    for epoch in range(epochs):
        if epoch % 10 == 0:  # Every 10 epochs
            writer.flush()
        model.train()
        loss = 0
        for batch_idx, (data, aug_data, target, length, fg, aug_fg, HELM, weight) in enumerate(
                tqdm(train_loader, desc="Training batches")):
            data = data.to(device)
            aug_data = aug_data.to(device) if aug_data[0] is not None else aug_data
            target = target.to(device)
            fg = fg.to(device)
            aug_fg = aug_fg.to(device) if aug_fg[0] is not None else aug_fg
            HELM = HELM.to(device) if HELM[0] is not None else HELM
            if text_optimizer is not None:
                text_optimizer.zero_grad()
            if other_optimizer is not None:
                other_optimizer.zero_grad()

            latent_feature, regression_logits, HELM_logits = model(data, fg)
            aug_latent_feature, aug_regression_logits, aug_HELM_logits = model(aug_data, aug_fg)

            train_loss = 0
            # 回归损失
            target = target.view(-1, 1)
            regression_loss = regression_criterion(regression_logits, target) + regression_criterion(
                aug_regression_logits,
                target)
            # HELM预测损失
            if model_config['use_HELM_loss']:
                # 计算每个任务的损失并累加
                HELM_losss = 0
                for i in range(HELM_logits.size(1)):
                    HELM_losss += HELM_criterion(HELM_logits[:, i], HELM[:, i]) + HELM_criterion(aug_HELM_logits[:, i],
                                                                                                 HELM[:, i])

                # 计算平均损失
                HELM_loss = HELM_losss / HELM_logits.size(1)
                HELM_loss = HELM_loss * 10
            else:
                HELM_loss = 0

            # 对比学习loss
            if model_config['use_CL_loss']:
                CL_loss = get_contrastiveLoss(latent_feature, aug_latent_feature, weight, model_config['CL_margin'])
            else:
                CL_loss = 0
            CL_loss = 50 * CL_loss
            train_loss = regression_loss + HELM_loss + CL_loss

            loss += train_loss.item()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)

            if text_optimizer is not None:
                text_optimizer.step()
            if other_optimizer is not None:
                other_optimizer.step()

            if hasattr(model, "use_ema") and model.use_ema:
                model.model_ema(model)

        loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}")
        writer.add_scalar('training loss', loss, epoch + 1)

        # Test
        model.eval()

        test_loss = 0
        all_y_true = []
        all_y_pred = []
        context_manager = model.ema_scope(context="ema version") if hasattr(model,
                                                                            'use_ema') and model.use_ema else nullcontext
        with context_manager and torch.no_grad():
            for batch_idx, (data, aug_data, target, length, fg, aug_fg, HELM, weight) in enumerate(
                    tqdm(test_loader, desc="Testing batches")):
                data = data.to(device)
                aug_data = aug_data.to(device) if aug_data[0] is not None else aug_data
                target = target.to(device)
                fg = fg.to(device)
                aug_fg = aug_fg.to(device) if aug_fg[0] is not None else aug_fg
                HELM = HELM.to(device) if HELM[0] is not None else HELM

                _, regression_logits, _ = model(data, fg)

                target = target.view(-1, 1).cpu().flatten()
                regression_logits = regression_logits.view(-1, 1).cpu().flatten()

                # R2
                all_y_true.append(target)
                all_y_pred.append(regression_logits)

        test_R2_score = r2_score(torch.cat(all_y_true), torch.cat(all_y_pred))
        test_loss = mse(torch.cat(all_y_true), torch.cat(all_y_pred))
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, R2_score: {test_R2_score:.4f}")
        writer.add_scalar('test loss', test_loss, epoch + 1)
        writer.add_scalar('test R2_score', test_R2_score, epoch + 1)

        val_loss = 0
        all_y_true = []
        all_y_pred = []
        with context_manager and torch.no_grad():
            for batch_idx, (data, aug_data, target, length,  fg, aug_fg, HELM, weight) in enumerate(
                    tqdm(val_loader, desc="Val batches")):
                data = data.to(device)
                aug_data = aug_data.to(device) if aug_data[0] is not None else aug_data
                target = target.to(device)
                fg = fg.to(device)
                aug_fg = aug_fg.to(device) if aug_fg[0] is not None else aug_fg
                HELM = HELM.to(device) if HELM[0] is not None else HELM

                _, regression_logits, _ = model(data, fg)

                target = target.view(-1, 1).cpu().flatten()
                regression_logits = regression_logits.view(-1, 1).cpu().flatten()

                # R2
                all_y_true.append(target)
                all_y_pred.append(regression_logits)
        val_R2_score = r2_score(torch.cat(all_y_true), torch.cat(all_y_pred))
        val_loss = mse(torch.cat(all_y_true), torch.cat(all_y_pred))
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, R2_score: {val_R2_score:.4f}")

        if (test_loss + 1 - test_R2_score) < best_test_loss:
            torch.save(model.state_dict(), os.path.join(os.path.splitext(log_file)[0], 'best_model.pth'))
            best_test_loss = test_loss + 1 - test_R2_score
            print("current best model: test_loss:{:.4f},R2_score:{:.4f}".format(test_loss, test_R2_score))


        writer.add_scalar('val loss', val_loss, epoch + 1)
        writer.add_scalar('val R2_score', val_R2_score, epoch + 1)

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}\n")
            f.write(f"Regression Loss: {regression_loss:.4f}, HELM Loss: {HELM_loss}, CL Loss: {CL_loss}\n")
            f.write(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, Test_R2_score: {test_R2_score: 4f}\n")
            f.write(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val_R2_score: {val_R2_score: 4f}\n")

        print("current best model:", best_test_loss)
