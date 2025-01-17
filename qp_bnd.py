import torch
import torch.nn as nn
import torch.optim as optim
from qpth.qp import QPFunction
import numpy as np
from sklearn.model_selection import train_test_split


class ThetaPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ThetaPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensure outputs are between 0 and 1
        )

    def forward(self, x):
        return self.net(x)


def gen_trans_probs_from_theta(theta, n_states, n_actions):
    num, dim = theta.shape
    assert dim == (n_states - 1) * n_actions
    theta_array = theta.reshape(num, n_states - 1, n_actions)
    trans_probs = torch.zeros(num, n_states, n_states, n_actions)
    for s in range(num):
        trans_probs[s, 0, 1, :] = 1
        trans_probs[s, torch.arange(1, n_states - 1), torch.arange(2, n_states), :] = theta_array[s, :(n_states - 2), :]
        trans_probs[s, -1, -1, :] = theta_array[s, -1, :]
        trans_probs[s, :, 0, :] = 1 - torch.sum(trans_probs[s, :, 1:n_states, :], dim=1)

    return trans_probs


# Generate synthetic data
def generate_qpdata_bnd(n, k, pow1, pow2, n_samples=500, xi_dim=2):
    # construct the state space
    states = range(1, n + 1)

    # construct the action set
    actions = range(1, k + 1)

    # Generate features and MDP paramters
    feats = torch.randn(n_samples, xi_dim)
    # Calculate the true parameters
    # Transforming the features to the c0 + c1 * i
    transform_matrix = torch.stack([torch.ones(n - 1), torch.arange(2, n + 1)])
    numerator = torch.repeat_interleave(torch.matmul(feats, transform_matrix), repeats=k, dim=1)
    denominator = torch.arange(1, k + 1).repeat(n - 1)
    thetas = 1 / (1 + torch.exp(numerator / denominator))  # Shape: (batch, (n-1)*k), flattened along the action dimension first

    # construct the cost function
    cost_fun = torch.exp((torch.arange(1, k + 1).reshape(1, -1)) ** pow1 / (torch.arange(1, n + 1).reshape(-1, 1)) ** pow2)

    # construct initial distribution
    gamma = 1 / n * torch.ones(n)

    return states, actions, feats, thetas, cost_fun, gamma


def solve_qp(theta_hat, theta_true, cost_fun, gamma, beta, epsilon, qp_solver):
    n, k = cost_fun.shape
    batch_size = theta_hat.shape[0]

    # Reconstruct transition probabilities from theta_hat
    trans_probs_hat = gen_trans_probs_from_theta(theta_hat, n, k)  # Shape: (batch, n, n, k)
    trans_probs_true = gen_trans_probs_from_theta(theta_true, n, k)  # Shape: (batch, n, n, k)

    # Objective: 0.5 z^T P z + q^T z
    # According to the problem: p^T z + epsilon z^T Q z
    # Map to QP standard form: P = 2 * epsilon * I, q = cost_fun.flatten()
    quad_coef = 2 * epsilon * torch.eye(n * k)
    lin_coef = cost_fun.view(-1)  # Shape: (n*k,)
    # Expand P and q for batch
    quad_coef = quad_coef.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch, n*k, n*k)
    lin_coef = lin_coef.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch, n*k)

    # Gz <= h (inequality constraints): y >= 0 => -I y <= 0
    ineq_coef = - torch.eye(n * k)
    ineq_rhs = torch.zeros(n * k)
    # Expand G and h for batch
    ineq_coef = ineq_coef.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, n*k, n*k)
    ineq_rhs = ineq_rhs.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, n*k)

    # Construct the equality coefficient array
    # A z = b, where A has shape (batch, n, n*k)
    eq_coef_hat = torch.eye(n).unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, k) - beta * trans_probs_hat  # Shape: (batch, n, n, k)
    eq_coef_hat = eq_coef_hat.permute(0, 2, 1, 3).reshape(batch_size, n, n * k)  # Shape: (batch, n, n*k)
    eq_coef_true = torch.eye(n).unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, k) - beta * trans_probs_true  # Shape: (batch, n, n, k)
    eq_coef_true = eq_coef_true.permute(0, 2, 1, 3).reshape(batch_size, n, n * k)  # Shape: (batch, n, n*k)
    eq_rhs = gamma.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch, n)

    # Solve QP: min 0.5 z^T P z + q^T z
    # Subject to Az = b and Gz <= h
    y_hat = qp_solver(quad_coef, lin_coef, ineq_coef, ineq_rhs, eq_coef_hat, eq_rhs)
    y_true = qp_solver(quad_coef, lin_coef, ineq_coef, ineq_rhs, eq_coef_true, eq_rhs)

    return y_hat, y_true


def train_eval(states, actions, feats, thetas, cost_fun, gamma, beta, epsilon=1e-2, xi_dim=2):
    # hyperparameters (to be tuned)
    batch_size = 20
    num_epochs = 100
    learning_rate = 1e-3

    n = len(states)  # Number of states
    k = len(actions)  # Number of actions

    # Split into training and test sets
    xi_train, xi_test, theta_train, theta_test = train_test_split(
        feats, thetas, test_size=0.2, random_state=42
    )

    # Initialize model, optimizer, and loss function
    model = ThetaPredictor(xi_dim, (n - 1) * k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss function
    loss_fn = nn.MSELoss(reduction='sum')

    # QPFunction expects specific formats
    qp_solver = QPFunction()

    # Training Loop
    num_batches = int(np.ceil(xi_train.shape[0] / batch_size))
    for epoch in range(num_epochs):
        model.train()
        train_size = xi_train.size(0)
        permutation = torch.randperm(train_size)
        xi_train_shuffled = xi_train[permutation]
        theta_train_shuffled = theta_train[permutation]

        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, train_size)

            xi_batch = xi_train_shuffled[start:end]  # Shape: (batch, xi_dim)
            theta_true = theta_train_shuffled[start:end]  # Shape: (batch, (n-1)*k)
            # Forward pass: predict theta_hat
            theta_hat = model(xi_batch)  # Shape: (batch, (n-1)*k)

            y_hat, y_true = solve_qp(theta_hat, theta_true, cost_fun, gamma, beta, epsilon, qp_solver)
            loss = loss_fn(y_hat, y_true)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / train_size
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_epoch_loss:.4f}")

    # Evaluation on Test Set
    model.eval()
    with torch.no_grad():
        test_size = xi_test.shape[0]
        num_test_batches = int(np.ceil(test_size / batch_size))
        test_loss = 0.0

        for batch_idx in range(num_test_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, test_size)

            xi_batch = xi_test[start:end]  # Shape: (batch, xi_dim)
            theta_true = theta_test[start:end]  # Shape: (batch, (n-1)*k)
            # Predict theta_hat
            theta_hat = model(xi_batch)  # Shape: (batch, (n-1)*k)

            y_hat, y_true = solve_qp(theta_hat, theta_true, cost_fun, gamma, beta, epsilon, qp_solver)

            # Compute test loss
            test_loss += loss_fn(y_hat, y_true).item()

        avg_test_loss = test_loss / test_size
        print(f"\nTest Loss: {avg_test_loss:.4f}")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    hyperparams = [{'lr': lr, 'hidden_dim': hd, 'num_epochs': ne}
                   for lr in [1e-2, 1e-3, 1e-4, 1e-5]
                   for hd in [32, 64, 128]
                   for ne in [200, 350, 500]]

    # generate the MDP of the input birth and death process
    S, A, xi, params, C, g = generate_qpdata_bnd(10, 5, 0.8, 1)
    # decision-focused learning
    train_eval(S, A, xi, params, C, g, 0.95, 1e-2)
