import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from mdp_lp.train_elasticity import occupancy_to_policy, eval_pi_on_true_mdp, eval_y_on_true_mdp
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.elasticity_model import ElasticityModel, LambdaPredictor

import matplotlib.pyplot as plt

config = {
    "s_max": 50,
    "b_max": 5,
    "q_max": 35,
    "d_max": 100,
    "foc": 0,
    "voc": 0.5,
    "hc": 0.1,
    "bc": 0,
    "beta": 0.95,
    "p_constant": 10.0,
    "p_lognormal_mean": 10.0,
    "p_lognormal_std": 0.5,
}

model_config = {
    "learning_rate": 0.001,
    "hidden_dim": 32,
    "use_p_related": False,
    "output_log": False,
}


def test_dfl_ope():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")

    pred_model = LambdaPredictor(model_config["hidden_dim"], model_config["use_p_related"]).to(device)
    # optimizer = optim.Adam(pred_model.parameters(), lr=model_config["learning_rate"])

    lambda_pred = torch.linspace(0, 100, 101, device=device)

    S = config["s_max"] + config["b_max"] + 1
    A = config["q_max"] + 1

    batch_size = lambda_pred.shape[0]
    lambda_true = torch.tensor([30] * batch_size, device=device)
    p = torch.ones(batch_size, device=device) * config["p_constant"]

    mdp = create_inventory_from_lambda(
            lambda_pred,
            config["b_max"], config["s_max"], config["q_max"], config["d_max"],
            config["foc"], config["voc"], config["hc"], config["bc"], config["beta"],
            p
        )
    H_use = mdp.build_H(mdp.P)
    r_flat_use = mdp.r.reshape(batch_size, -1)
    gamma_use = mdp.build_gamma(batch_size, S, device)

    y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_use)
    
    with torch.no_grad():
        pi, _ = occupancy_to_policy(y_star, S, A)
        mdp_true = create_inventory_from_lambda(
            lambda_true,
            config["b_max"], config["s_max"], config["q_max"], config["d_max"],
            config["foc"], config["voc"], config["hc"], config["bc"], config["beta"],
            p
        )
        v_star, _y_pi_true = eval_pi_on_true_mdp(mdp_true, pi, gamma_use)
        H_true = mdp_true.build_H(mdp_true.P)
        check_eq = H_true @ _y_pi_true.reshape(batch_size, -1, 1) - gamma_use.unsqueeze(-1)
        print(f"check_eq: {check_eq.abs().max()}")
        # v_star, _y_pi_true = eval_y_on_true_mdp(mdp_true, y_star, gamma_use)
    return lambda_pred.tolist(), v_star.tolist(), lambda_true[0].item()

if __name__ == "__main__":
    lambda_pred, v_star, lambda_true_val = test_dfl_ope()
    # print(f"v_star: {v_star}")
    
    plt.plot(lambda_pred, v_star)
    plt.axvline(x=lambda_true_val, color="r", linestyle="--", label=f"lambda_true={lambda_true_val}")
    plt.xlabel("lambda")
    plt.ylabel("v_star")
    plt.title("v_star vs lambda")
    plt.legend()
    plt.savefig("v_star_vs_lambda.pdf", dpi=300, bbox_inches="tight")
    plt.show()