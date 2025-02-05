from dflsa_train import *


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
    batch_size = 20

    feats = torch.rand(n_samples, n_feats)
    mdp = BirthDeath(n, k, feats, pow1, pow2, beta, partial(expon_mdl, n=n, k=k))
    mdp_dataset = TensorDataset(feats, mdp.trans_probs, mdp.obj_fun, mdp.init_dist)
    params = {'states': mdp.states, 'actions': mdp.actions, 'beta': mdp.disc_factor, 'pow1': mdp.pow1, 'pow2': mdp.pow2,
              'n_feats': n_feats, 'num_epochs': num_epochs}
    config = {'lr': 5e-4, 'hidden_dim': 128, 'quad_coefs': 5}

    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    train_dataset, test_dataset = random_split(mdp_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_result = train_model(config, params, train_loader, test_loader, d)

    # Save result to a file
    result_file = Path(f'test_result.json')
    result_file.write_text(json.dumps(test_result, indent=4))


if __name__ == '__main__':
    main()
