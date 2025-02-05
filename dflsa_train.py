import json
import os
import torch.optim as optim
from functools import partial
from mdp import *
from utils import *
from qpth.qp import QPFunction
from state_aggregation_alg import state_agg
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


def solve_qp(mdp, quad_coef, qp_solver):
    start_time = time.time()

    trans_probs = mdp.trans_probs
    obj_fun = mdp.obj_fun
    gamma = mdp.init_dist
    beta = mdp.disc_factor
    n_samples, n, k = obj_fun.shape

    # scale the problem for numerical stability
    scale = 1 / torch.max(torch.ones_like(obj_fun), torch.abs(obj_fun)).reshape(n_samples, -1)  # Shape: (n_samples, n*k)

    # Objective: 0.5 z^T P z + q^T z
    # P = 2 * quad_coef * I
    # q = flattened obj_fun
    # quad_mat = 2 * quad_coef * torch.eye(n * k).unsqueeze(0).repeat(n_samples, 1, 1)  # Shape: (n_samples, n*k, n*k)
    quad_mat = 2 * quad_coef * torch.diag_embed(scale ** 2)
    lin_coef = obj_fun.reshape(n_samples, -1) * scale  # Shape: (n_samples, n*k)

    # Gz <= h (inequality constraints): y >= 0 => -I y <= 0
    # ineq_coef = - torch.eye(n * k).unsqueeze(0).repeat(n_samples, 1, 1)  # Shape: (n_samples, n*k, n*k)
    ineq_coef = - torch.diag_embed(scale)
    ineq_rhs = torch.zeros(n_samples, n * k)  # Shape: (n_samples, n*k)

    # Construct the equality coefficient array
    # A z = b, where A has shape (n_samples, n, n*k)
    eq_coef = torch.eye(n).unsqueeze(0).unsqueeze(3).repeat(n_samples, 1, 1, k) - beta * trans_probs  # Shape: (n_samples, n, n, k)
    # permute dimensions 1 and 2 because we need the constraint for each j (dim 2)
    eq_coef = eq_coef.permute(0, 2, 1, 3).reshape(n_samples, n, n * k) * scale.unsqueeze(1)  # Shape: (n_samples, n, n*k)
    eq_rhs = gamma  # Shape: (n_samples, n)t

    # Solve QP: min 0.5 z^T P z + q^T z
    # Subject to Az = b and Gz <= h
    y = qp_solver(quad_mat, lin_coef, ineq_coef, ineq_rhs, eq_coef, eq_rhs) * scale
    obj = torch.sum(lin_coef * y / scale, dim=1)

    run_time = time.time() - start_time

    return y, obj, run_time


def train_model(config, params, train_loader, eval_loader, d):
    learning_rate = config['lr']
    hidden_dim = config['hidden_dim']
    quad_coef = config['quad_coefs']

    states = params['states']
    actions = params['actions']
    beta = params['beta']
    pow1 = params['pow1']
    pow2 = params['pow2']
    n_feats = params['n_feats']
    num_epochs = params['num_epochs']
    n = len(states)
    k = len(actions)

    loss_fn = nn.MSELoss()
    qp_solver = QPFunction()

    model = ThetaPredictor(n_feats, (n - 1) * k, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for xi_batch, p_batch, c_batch, gamma_batch in train_loader:
            # prepare a batch of true and estimated MDPs
            mdp_true_batch = MDP(states, actions, p_batch, c_batch, gamma_batch, beta)
            mdp_hat_batch = BirthDeath(n, k, xi_batch, pow1, pow2, beta, model)
            # apply state aggregation for the estimated MDPs
            mdp_hat_agg = state_agg(mdp_hat_batch, d)[0]
            obj_hat = solve_qp(mdp_hat_agg, quad_coef, qp_solver)[1]

            obj_true = mdp_true_batch.policy_iteration()[1]
            loss = loss_fn(obj_hat, obj_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

    model.eval()
    eval_run_time = 0.0
    eval_loss = 0.0
    with torch.no_grad():
        for xi_batch, p_batch, c_batch, gamma_batch in eval_loader:
            mdp_true_batch = MDP(states, actions, p_batch, c_batch, gamma_batch, beta)
            start_time = time.time()
            mdp_hat_batch = BirthDeath(n, k, xi_batch, pow1, pow2, beta, model)
            mdp_hat_agg = state_agg(mdp_hat_batch, d)[0]
            obj_hat = solve_qp(mdp_hat_agg, quad_coef, qp_solver)[1]
            eval_run_time += time.time() - start_time
            obj_true = mdp_true_batch.policy_iteration()[1]
            eval_loss += loss_fn(obj_hat, obj_true).item()

    avg_eval_loss = eval_loss / len(eval_loader)
    print(f'Evaluation Loss: {avg_eval_loss:.4f}')
    print(f'Average Run Time for DFLSA: {eval_run_time / len(eval_loader.dataset):.4f}')

    return {'config': config, 'eval_loss': avg_eval_loss}


def main():
    set_seed(42)
    # generate the MDP of the input birth and death process
    n = 500  # number of states: |S|
    k = 20  # number of actions: |A|
    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor
    d = 5  # the number of states in the cluster set
    n_samples = 100
    n_feats = 2
    num_epochs = 100
    batch_size = 16

    feats = torch.rand(n_samples, n_feats)
    mdp = BirthDeath(n, k, feats, pow1, pow2, beta, partial(expon_mdl, n=n, k=k))
    mdp_dataset = TensorDataset(feats, mdp.trans_probs, mdp.obj_fun, mdp.init_dist)
    params = {'states': mdp.states, 'actions': mdp.actions, 'beta': mdp.disc_factor, 'pow1': mdp.pow1, 'pow2': mdp.pow2,
              'n_feats': n_feats, 'num_epochs': num_epochs}

    # Get the SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    # Define hyperparameter configurations
    hyperparams = [{'lr': lr, 'hidden_dim': hd, 'quad_coefs': qc}
                   for lr in [1e-3, 5e-4, 1e-4, 5e-5]
                   for hd in [32, 64, 128]
                   for qc in [1, 2, 5, 10]]

    # Select the hyperparameter configuration for this task
    config = hyperparams[task_id]

    train_size = int(0.8 * n_samples)
    val_size = int(0.2 * train_size)
    train_tune_size = train_size - val_size
    test_size = n_samples - train_size
    train_dataset, test_dataset = random_split(mdp_dataset, [train_size, test_size])
    train_tune_dataset, val_dataset = random_split(train_dataset, [train_tune_size, val_size])
    train_tune_loader = DataLoader(train_tune_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    tune_result = train_model(config, params, train_tune_loader, val_loader, d)

    # Save result to a file
    result_file = Path(f'result_{task_id}.json')
    result_file.write_text(json.dumps(tune_result, indent=4))


if __name__ == '__main__':
    main()
