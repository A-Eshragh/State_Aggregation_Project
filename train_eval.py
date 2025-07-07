import time
import torch
import torch.nn as nn
import torch.optim as optim
from mdp_bnd import BirthDeath
from sa_sparse import state_agg
from utils import GroundTruth, ThetaPredictor, solve_qp
from qpth.qp import QPFunction
from torch.utils.data import DataLoader, TensorDataset, random_split


def train_eval(params, hyperparam, train_loader, eval_loader):
    n = params['n_states']
    k = params['n_actions']
    m = params['n_neighbors']
    two_way = params['two_way']
    d = params['cluster_size']
    algo = params['algo']
    pow1 = params['pow1']
    pow2 = params['pow2']
    beta = params['beta']
    n_feats = params['n_feats']
    num_epochs = params['num_epochs']
    reg_coef = params['reg_coef']
    loss_weight = params['loss_weight']

    learning_rate = hyperparam['lr']
    hidden_dim = hyperparam['hidden_dim']

    max_nnz_per_row = 2 * (m + 1) if two_way else m + 1
    gt_mdl = GroundTruth(n_feats, n * max_nnz_per_row * k)
    pred_model = ThetaPredictor(n_feats, hidden_dim, n * max_nnz_per_row * k)
    loss_fn = nn.L1Loss()
    qp_solver = QPFunction()
    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)

    train_time_list = []
    obj_loss_list = []  # list of optimality gaps
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = 0.0
        obj_loss = 0.0
        pred_model.train()
        for feats, obj_true in train_loader:
            theta_hat = pred_model(feats)
            theta_true = gt_mdl(feats)
            pred_loss = loss_fn(theta_hat, theta_true)
            mdp_hat = BirthDeath(n, k, m, two_way, theta_hat, pow1, pow2, beta)
            if d == -1:
                # in this case, twostage method with policy iteration will be used
                obj_hat = torch.from_numpy(mdp_hat.policy_iteration()[1]).float()  # gradient not required
                obj_diff = loss_fn(obj_hat, obj_true)
                loss = pred_loss
            else:
                mdp_hat_agg = state_agg(mdp_hat, d)[0]
                obj_hat = solve_qp(mdp_hat_agg, reg_coef, qp_solver)[1]
                obj_diff = loss_fn(obj_hat, obj_true)
                loss = pred_loss if algo == 'twostage' else obj_diff + loss_weight * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            obj_loss += obj_diff.item()

        train_time_list.append(time.time() - start_time)
        avg_train_loss = train_loss / len(train_loader)
        avg_obj_loss = obj_loss / len(train_loader)
        obj_loss_list.append(avg_obj_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Objective Loss: {avg_obj_loss:.4f}')

    pred_model.eval()
    pred_loss = 0.0
    eval_loss = 0.0
    true_values = 0.0
    start_time = time.time()
    with torch.no_grad():
        for feats, obj_true in eval_loader:
            theta_hat = pred_model(feats)
            theta_true = gt_mdl(feats)
            pred_loss += loss_fn(theta_hat, theta_true).item()
            mdp_hat = BirthDeath(n, k, m, two_way, theta_hat, pow1, pow2, beta)
            if d == -1:
                _, obj_hat, pi_time = mdp_hat.policy_iteration()
                print(f'Policy Iteration Runtime: {pi_time / len(obj_true):.4f}s')
                obj_hat = torch.from_numpy(obj_hat).float()
            else:
                mdp_hat_agg, sa_time = state_agg(mdp_hat, d)
                print(f'State Aggregation Runtime: {sa_time / len(obj_true):.4f}s')
                _, obj_hat, qp_time = solve_qp(mdp_hat_agg, reg_coef, qp_solver)
                print(f'Quadratic Programming Runtime: {qp_time / len(obj_true):.4f}s')

            eval_loss += loss_fn(obj_hat, obj_true).item()  # average over the batch by default
            true_values += torch.mean(torch.abs(obj_true)).item()

    eval_time = time.time() - start_time
    eval_time /= len(eval_loader.dataset)
    pred_loss /= len(eval_loader)
    eval_loss /= len(eval_loader)
    true_values /= len(eval_loader)
    eval_loss /= true_values  # this is essentially the WAPE
    print(f'Average Runtime: {eval_time:.4f}s')
    print(f'Evaluation Loss: {eval_loss * 100:.2f}%')

    return {'hyperparam': hyperparam, 'params': params, 'eval_time': eval_time, 'pred_loss': pred_loss,
            'eval_loss': eval_loss, 'obj_loss': obj_loss_list, 'train_time': train_time_list}


def prepare_data(params, mode):
    n = params['n_states']
    k = params['n_actions']
    m = params['n_neighbors']
    two_way = params['two_way']
    pow1 = params['pow1']
    pow2 = params['pow2']
    beta = params['beta']
    n_samples = params['n_samples']
    n_feats = params['n_feats']
    batch_size = params['batch_size']

    feats_clean = torch.distributions.Cauchy(loc=0.0, scale=1.0).sample(torch.Size([n_samples, n_feats]))
    feats_clean = torch.clamp(feats_clean, min=-50, max=50)  # clip the feature values for numerical stability of neural networks
    max_nnz_per_row = 2 * (m + 1) if two_way else m + 1
    gt_mdl = GroundTruth(n_feats, n * max_nnz_per_row * k)
    thetas = gt_mdl(feats_clean)
    mdp_true = BirthDeath(n, k, m, two_way, thetas, pow1, pow2, beta)
    obj_true = torch.from_numpy(mdp_true.policy_iteration()[1]).float()

    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    # the learning agent only has access to noisy features
    noise = torch.randn_like(feats_clean) * feats_clean.std(dim=0, unbiased=False) / 3  # signal-to-noise ratio 3:1
    feats = feats_clean + noise
    train_dataset, test_dataset = random_split(TensorDataset(feats, obj_true), [train_size, test_size])
    if mode == 'train':
        train_tune_size = int(0.8 * train_size)
        val_size = train_size - train_tune_size
        train_dataset, test_dataset = random_split(train_dataset, [train_tune_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
