# Sparse LP solver using COO format

import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import coo_matrix


class SparseLPSolver:
    """
    Sparse LP solver using Gurobi with COO (Coordinate) format.
    Solves the discounted MDP LP for each batch element:
    
        maximize     r^T y
        subject to   H y = gamma
                     y >= 0
    
    Uses sparse matrix representation for H to handle large problems efficiently.
    
    Inputs:
        H:      (batch, S, SA) - can be dense or sparse
        r:      (batch, SA)
        gamma:  (batch, S)
    
    Outputs:
        y_star: (batch, SA)
        basis:  (batch, S)   # basis column indices
    """
    
    def __init__(self):
        pass
    
    def solve_single(self, H_i, r_i, gamma_i):
        """
        Solve ONE LP using Gurobi with sparse COO format.
        
        H_i:     (S, SA) - numpy array (will be converted to COO)
        r_i:     (SA,) - numpy array
        gamma_i: (S,) - numpy array
        """
        S, SA = H_i.shape
        
        # Convert H to COO format for sparse representation
        H_coo = coo_matrix(H_i)
        
        model = gp.Model()
        model.Params.OutputFlag = 0  # silence logs
        
        # Variables y_j >= 0
        y = model.addMVar(shape=SA, lb=0.0, vtype=GRB.CONTINUOUS)
        
        # Objective: maximize r^T y
        model.setObjective(r_i @ y, GRB.MAXIMIZE)
        
        # Equality constraints: H y = gamma
        # Build constraints using COO format efficiently
        # For each row in H, add constraint: sum(H[row, :] * y) = gamma[row]
        for row_idx in range(S):
            # Get all non-zero entries in this row
            row_mask = H_coo.row == row_idx
            if np.any(row_mask):
                # Get column indices and values for this row
                cols = H_coo.col[row_mask]
                vals = H_coo.data[row_mask]
                
                # Build constraint: sum(vals[j] * y[cols[j]]) = gamma[row_idx]
                expr = gp.LinExpr()
                for j, col_idx in enumerate(cols):
                    expr += vals[j] * y[col_idx]
                
                model.addConstr(expr == gamma_i[row_idx])
            else:
                # Empty row: 0 = gamma[row_idx]
                model.addConstr(0 == gamma_i[row_idx])
        
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi failed to solve LP.")
        
        # Extract y*
        y_star = y.X  # numpy array
        
        # Extract basis:
        # VBasis[j] == 0  → basic variable
        # VBasis[j] != 0 → nonbasic
        vb = y.VBasis
        
        # There should be EXACTLY S basic vars (square basis matrix)
        basis_idx = torch.tensor(
            [j for j in range(SA) if vb[j] == 0][:S],
            dtype=torch.long
        )
        
        return torch.tensor(y_star, dtype=torch.float32), basis_idx
    
    def solve(self, H, r, gamma):
        """
        Solve LP for the whole batch using sparse format.
        
        H:      (batch, S, SA) - PyTorch tensor
        r:      (batch, SA) - PyTorch tensor
        gamma:  (batch, S) - PyTorch tensor
        
        Returns:
            y_star: (batch, SA)
            basis:  (batch, S)
        """
        batch, S, SA = H.shape
        
        ys = []
        bases = []
        
        for b in range(batch):
            # Convert to numpy and use COO format
            H_b = H[b].cpu().numpy()
            r_b = r[b].cpu().numpy()
            gamma_b = gamma[b].cpu().numpy()
            
            y_b, B_b = self.solve_single(H_b, r_b, gamma_b)
            ys.append(y_b)
            bases.append(B_b)
        
        return torch.stack(ys, dim=0), torch.stack(bases, dim=0)


def solve_lp_sparse(H, r, gamma):
    """
    Solve LP using sparse COO format for single instance.
    
    H:     (S, SA) or (batch, S, SA) - PyTorch tensor
    r:     (SA,) or (batch, SA) - PyTorch tensor
    gamma: (S,) or (batch, S) - PyTorch tensor
    
    Returns:
        y_star: (SA,) or (batch, SA)
        basis:  (S,) or (batch, S)
    """
    solver = SparseLPSolver()
    
    # Handle both single and batched inputs
    if H.dim() == 3:
        # Batched
        y_star, basis = solver.solve(H, r, gamma)
    else:
        # Single instance
        y_star, basis = solver.solve_single(
            H.cpu().numpy(),
            r.cpu().numpy(),
            gamma.cpu().numpy()
        )
    
    return y_star, basis
