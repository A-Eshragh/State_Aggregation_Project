import time
import torch
import torch.nn as nn
import torch.optim as optim
from mdp_derived import BirthDeath, Inventory
# from sa_sparse import state_agg
from sa_sparse import SoftAgg
from utils import GroundTruth, ThetaPredictor, solve_qp
from qpth.qp import QPFunction, QPSolvers
from torch.utils.data import DataLoader, TensorDataset, random_split


MDP_CLASSES = {
    'bnd': lambda args, theta_hat: BirthDeath(
        args.n_states, args.n_actions, args.n_neighbors, args.two_way,
        args.pow1, args.pow2, args.beta, theta_hat
    ),
    'inventory': lambda args, theta_hat: Inventory(
        args.b_max, args.s_max, args.q_max, args.d_max, args.foc, args.voc,
        args.hc, args.bc, args.beta, theta_hat
    ),
}


def train_eval(args, train_loader, eval_loader):
    hyperparam = args.hyperparam
    learning_rate = hyperparam['lr']
    hidden_dim = hyperparam['hidden_dim']

    gt_mdl = GroundTruth(args.n_feats, args.n_params)
    pred_model = ThetaPredictor(args.n_feats, hidden_dim, args.n_params)
    params = list(pred_model.parameters())
    agg = None
    if args.n_agg != -1:
        n_states = args.n_states if args.mdp_type == 'bnd' else args.b_max + args.s_max + 1
        n_actions = args.n_actions if args.mdp_type == 'bnd' else args.q_max + 1
        agg = SoftAgg(n_states, n_actions, args.n_agg + 1)
        params = params + list(agg.parameters())

    optimizer = optim.Adam(params, lr=learning_rate)
    loss_fn = nn.L1Loss()
    qp_solver = QPFunction(solver=QPSolvers.CVXPY)

    train_time_list = []
    obj_loss_list = []  # list of optimality gaps
    print(f"[Training] Starting training: {args.n_epochs} epochs, algo={args.algo}, n_agg={args.n_agg}")
    start_time = time.time()
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        obj_loss = 0.0
        pred_model.train()
        for feats, obj_true in train_loader:
            # 1) Predict parameters and supervise with ground truth
            theta_hat = pred_model(feats)
            theta_true = gt_mdl(feats)
            pred_loss = loss_fn(theta_hat, theta_true)

            # 2) Build the MDP (unaggregated)
            try:
                mdp_hat = MDP_CLASSES[args.mdp_type](args, theta_hat)
            except KeyError:
                raise ValueError(f'No such MDP type: {args.mdp_type}')

            # 3) Aggregate if requested
            mdp_working = mdp_hat if args.n_agg == -1 else agg(mdp_hat)[0]

            # 4) Decide how to solve for the objective prediction (obj_hat)
            have_obj = False
            obj_hat = None
            obj_diff = None

            if args.algo == 'dfl':
                # Always use QP solver under DFL, regardless of aggregation
                obj_hat = solve_qp(mdp_working, args.reg_coef, qp_solver)[1]
                have_obj = True
            elif args.algo == 'twostage':
                # Use policy iteration with the special-case gate:
                #   only run policy iteration for the *unaggregated* problem if train_log is True
                if (args.n_agg == -1 and args.train_log) or (args.n_agg != -1):
                    obj_hat = torch.from_numpy(mdp_working.policy_iteration()[1]).float()
                    have_obj = True
            else:
                raise ValueError(f"Unsupported args.algo='{args.algo}'. Expected 'dfl' or 'twostage'.")

            # 5) If we produced an objective prediction, compute obj_diff
            if have_obj:
                obj_diff = loss_fn(obj_hat, obj_true)
                obj_loss += obj_diff.item()

            # 6) Compose the training loss per algorithm spec
            if args.algo == 'twostage':
                # Two-stage always trains on prediction loss only
                loss = pred_loss
            else:
                # DFL combines downstream objective error plus weighted prediction loss
                loss = obj_diff + args.loss_weight * pred_loss

            # 7) Backprop, (optionally) clip grads, step
            optimizer.zero_grad()
            loss.backward()

            # Log (and optionally clip) gradient norm
            # total_norm = torch.norm(torch.stack([
            #     p.grad.detach().norm(2) for p in pred_model.parameters() if p.grad is not None
            # ]))
            # print(f'Epoch {epoch}: grad norm = {total_norm:.4e}')

            # Example (optional): clip if you want to enforce a ceiling
            # torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=10.0)

            optimizer.step()
            train_loss += loss.item()

        if args.train_log:
            train_time_list.append(time.time() - start_time)
            avg_obj_loss = obj_loss / len(train_loader)
            obj_loss_list.append(avg_obj_loss)

        avg_train_loss = train_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} / {args.n_epochs}, Training Loss: {avg_train_loss:.4f}')

    print(f"[Training] Training complete. Starting evaluation...")
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
            try:
                mdp_hat = MDP_CLASSES[args.mdp_type](args, theta_hat)
            except KeyError:
                raise ValueError(f'No such MDP type: {args.mdp_type}')

            # print_weights_and_policy(mdp_hat, agg)
            if args.n_agg == -1:
                mdp_working = mdp_hat
            else:
                # mdp_hat_agg, sa_time = state_agg(mdp_hat, args.n_agg)
                mdp_working, sa_time = agg(mdp_hat)
                print(f'State Aggregation Runtime: {sa_time / len(obj_true):.4f}s')

            if args.algo == 'dfl':
                _, obj_hat, qp_time = solve_qp(mdp_working, args.reg_coef, qp_solver)
                print(f'Quadratic Programming Runtime: {qp_time / len(obj_true):.4f}s')
            else:
                _, obj_hat, pi_time = mdp_working.policy_iteration()
                print(f'Policy Iteration Runtime: {pi_time / len(obj_true):.4f}s')
                obj_hat = torch.from_numpy(obj_hat).float()

            eval_loss += loss_fn(obj_hat, obj_true).item()  # average over the batch by default
            true_values += torch.mean(torch.abs(obj_true)).item()

    eval_time = time.time() - start_time
    eval_time /= len(eval_loader.dataset)
    pred_loss /= len(eval_loader)
    eval_loss /= len(eval_loader)
    true_values /= len(eval_loader)
    eval_loss /= true_values  # this is essentially the WAPE
    print(f'[Evaluation] Complete!')
    print(f'Average Runtime: {eval_time:.4f}s')
    print(f'Evaluation Loss: {eval_loss * 100:.2f}%')

    return vars(args) | {'eval_time': eval_time, 'pred_loss': pred_loss, 'eval_loss': eval_loss,
                         'obj_loss': obj_loss_list, 'train_time': train_time_list}


def prepare_data(args):
    print(f"[Data Prep] Generating {args.n_samples} feature vectors...")
    feats_clean = torch.distributions.Cauchy(loc=0.0, scale=1.0).sample(torch.Size([args.n_samples, args.n_feats]))
    # clip the feature values for numerical stability of neural networks
    feats_clean = torch.clamp(feats_clean, min=-50, max=50)
    print(f"[Data Prep] Features generated. Creating MDPs...")
    mdp_true = None
    match args.mdp_type:
        case 'bnd':
            max_nnz_per_row = 2 * args.n_neighbors + 1 if args.two_way else args.n_neighbors + 1
            args.n_params = args.n_states * args.n_actions * max_nnz_per_row
            gt_mdl = GroundTruth(args.n_feats, args.n_params)
            mdp_true = BirthDeath(args.n_states, args.n_actions, args.n_neighbors, args.two_way,
                                  args.pow1, args.pow2, args.beta, gt_mdl(feats_clean))
        case 'inventory':
            args.n_params = args.d_max + 1
            gt_mdl = GroundTruth(args.n_feats, args.n_params)
            mdp_true = Inventory(args.b_max, args.s_max, args.q_max, args.d_max, args.foc, args.voc,
                                 args.hc, args.bc, args.beta, gt_mdl(feats_clean))
        case _:
            raise ValueError('No such MDP type!')
    print(f"[Data Prep] MDPs created. Solving {args.n_samples} MDPs with policy iteration (this may take a while)...")
    obj_true = torch.from_numpy(mdp_true.policy_iteration()[1]).float()
    print(f"[Data Prep] All MDPs solved. Creating data loaders...")

    train_size = int(0.8 * args.n_samples)
    test_size = args.n_samples - train_size
    # the learning agent only has access to noisy features
    noise = torch.randn_like(feats_clean) * feats_clean.std(dim=0, unbiased=False) / 3  # signal-to-noise ratio 3:1
    feats = feats_clean + noise
    train_dataset, test_dataset = random_split(TensorDataset(feats, obj_true), [train_size, test_size])
    if args.mode == 'train':
        train_tune_size = int(0.8 * train_size)
        val_size = train_size - train_tune_size
        train_dataset, test_dataset = random_split(train_dataset, [train_tune_size, val_size])
    batch_size = int(0.8 * 0.2 * args.n_samples) if args.mode == 'train' else int(0.2 * args.n_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def print_weights_and_policy(mdp, agg, tau=0.5):
    n_samples = mdp.trans_probs.size(0)
    n, k, r = agg.n, agg.k, agg.r

    sp = mdp.trans_probs.coalesce()
    idx = sp.indices()  # (3, nnz): [s, row, col]
    val = sp.values()  # (nnz,)

    embed = agg.dest_embed.weight
    embed_cols = embed[idx[2], :]  # (nnz, r)
    contrib_phi = embed_cols * val.unsqueeze(1)  # P @ E: (nnz, r)
    phi_flat = torch.zeros(n_samples * n * k, r)  # (S*nk, r)
    keys_sia = idx[0] * n * k + idx[1]  # linear index of (s, ia): (nnz,)
    phi_flat.index_add_(0, keys_sia, contrib_phi)  # sum over the destination states
    phi = phi_flat.view(n_samples, n, k * r)  # (S, n, k*r)

    logits = agg.to_logits(phi)  # (S, n, m)
    weight = torch.softmax(logits / tau, dim=2)  # (S, n, m)
    print('The soft memberships for the first MDP instance are:')
    print(weight[0])

    policy_hat, obj_hat, pi_time = mdp.policy_iteration()
    print('The optimal policy of the predicted MDP is:')
    print(policy_hat)
    print('The predicted objective is:')
    print(obj_hat)
    print(f'The computation time is {pi_time:.2f}s.')
