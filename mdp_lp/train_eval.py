import time
import torch
import torch.nn as nn
import torch.optim as optim
from mdp_lp.mdp_derived import BirthDeathLP, InventoryLP
from mdp_lp.utils import GroundTruth, ThetaPredictor, solve_lp
# Note: Solvers are imported dynamically based on solver_type
from torch.utils.data import DataLoader, TensorDataset, random_split


MDP_CLASSES = {
    'bnd': lambda args, theta_hat: BirthDeathLP(
        args.n_states, args.n_actions, args.n_neighbors, args.two_way,
        args.pow1, args.pow2, args.beta, theta_hat
    ),
    'inventory': lambda args, theta_hat: InventoryLP(
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
    
    # Use dense LP solver
    from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        y, basis = solver_obj.solve(H, r, gamma)
        return y, basis

    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    train_time_list = []
    obj_loss_list = []  # list of optimality gaps
    print(f"[Training] Starting training: {args.n_epochs} epochs, algo={args.algo}, solver=dense")
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

            # 2) Build the MDP (LP formulation)
            try:
                mdp_hat = MDP_CLASSES[args.mdp_type](args, theta_hat)
            except KeyError:
                raise ValueError(f'No such MDP type: {args.mdp_type}')

            # 3) Decide how to solve for the objective prediction (obj_hat)
            have_obj = False
            obj_hat = None
            obj_diff = None

            if args.algo == 'dfl':
                # Use LP solver under DFL
                _, obj_hat, _ = solve_lp(mdp_hat, lp_solver)
                have_obj = True
            elif args.algo == 'twostage':
                # Two-stage: only train on prediction loss
                # But always compute obj_loss for tracking (even if not used in loss)
                _, obj_hat, _ = solve_lp(mdp_hat, lp_solver)
                obj_hat = obj_hat.detach()
                have_obj = True
            else:
                raise ValueError(f"Unsupported args.algo='{args.algo}'. Expected 'dfl' or 'twostage'.")

            # 4) If we produced an objective prediction, compute obj_diff
            if have_obj:
                obj_diff = loss_fn(obj_hat, obj_true)
                obj_loss += obj_diff.item()

            # 5) Compose the training loss per algorithm spec
            if args.algo == 'twostage':
                # Two-stage always trains on prediction loss only
                loss = pred_loss
            else:
                # DFL combines downstream objective error plus weighted prediction loss
                loss = obj_diff + args.loss_weight * pred_loss

            # 6) Backprop and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Always track obj_loss and train_time (not just when train_log is True)
        train_time_list.append(time.time() - start_time)
        avg_obj_loss = obj_loss / len(train_loader) if len(train_loader) > 0 else 0.0
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

            # Solve LP for evaluation
            _, obj_hat, lp_time = solve_lp(mdp_hat, lp_solver)
            print(f'Linear Programming Runtime: {lp_time / len(obj_true):.4f}s')

            eval_loss += loss_fn(obj_hat, obj_true).item()
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
            mdp_true = BirthDeathLP(args.n_states, args.n_actions, args.n_neighbors, args.two_way,
                                     args.pow1, args.pow2, args.beta, gt_mdl(feats_clean))
        case 'inventory':
            args.n_params = args.d_max + 1
            gt_mdl = GroundTruth(args.n_feats, args.n_params)
            mdp_true = InventoryLP(args.b_max, args.s_max, args.q_max, args.d_max, args.foc, args.voc,
                                    args.hc, args.bc, args.beta, gt_mdl(feats_clean))
        case _:
            raise ValueError('No such MDP type!')
    
    print(f"[Data Prep] MDPs created. Solving {args.n_samples} MDPs with LP solver (this may take a while)...")
    # Solve using LP to get true optimal values
    from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        y, basis = solver_obj.solve(H, r, gamma)
        return y, basis
    _, obj_true, _ = solve_lp(mdp_true, lp_solver)
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
