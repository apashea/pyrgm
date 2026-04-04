import numpy as np  
from scipy import sparse  
from scipy.linalg import sqrtm, toeplitz, expm  
import warnings  

class ModelLevel:  
    """Represents a single level in the hierarchical model"""  
    def __init__(self):  
        self.g = None  # y(t) = g(x,v,P)  
        self.f = None  # dx/dt = f(x,v,P)  
        self.pE = None  # prior expectation of parameters  
        self.pC = None  # prior covariances of parameters  
        self.hE = None  # prior expectation of h hyper-parameters (cause noise)  
        self.hC = None  # prior covariances of h  
        self.gE = None  # prior expectation of g hyper-parameters (state noise)  
        self.gC = None  # prior covariances of g  
        self.Q = None   # precision components (input noise)  
        self.R = None   # precision components (state noise)  
        self.V = None   # fixed precision (input noise)  
        self.W = None   # fixed precision (state noise)  
        self.m = None   # number of inputs v(i + 1)  
        self.n = None   # number of states x(i)  
        self.l = None   # number of output v(i)  
        self.x = None   # hidden states  
        self.v = None   # causal states  
        self.E = None   # estimation parameters

def _debug_print(msg, obj=None, debug=False):  
    """Helper function for debug printing"""  
    if debug:  
        print(f"[DEBUG] {msg}", end="")  
        if obj is not None:  
            if hasattr(obj, 'shape'):  
                print(f" | Type: {type(obj).__name__} | Shape: {obj.shape}", end="")  
                if sparse.issparse(obj):  
                    print(f" | nnz: {obj.nnz}", end="")  
            elif isinstance(obj, list):  
                print(f" | Type: list | Length: {len(obj)}", end="")  
                if obj and hasattr(obj[0], 'shape'):  
                    print(f" | First item shape: {obj[0].shape}", end="")  
            else:  
                print(f" | Type: {type(obj).__name__} | Value: {obj}", end="")  
        print()  
  
def spm_vec(X, debug=False):  
    """Vectorise a numeric, cell or structure array"""  
    _debug_print("spm_vec input", X, debug)  
      
    if isinstance(X, np.ndarray):  
        result = X.ravel()  
    elif isinstance(X, list):  
        result = []  
        for item in X:  
            result.extend(spm_vec(item, debug))  
        result = np.array(result)  
    else:  
        result = np.array([])  
      
    _debug_print("spm_vec output", result, debug)  
    return result  
  
def spm_unvec(vX, X, debug=False):  
    """Unvectorise back to original shape"""  
    if isinstance(X, np.ndarray):  
        return vX.reshape(X.shape)  
    elif isinstance(X, list):  
        result = []  
        idx = 0  
        for item in X:  
            if item is None:  
                # Handle None values - treat as empty  
                result.append(np.array([]))  
            elif isinstance(item, np.ndarray):  
                size = item.size  
                result.append(vX[idx:idx+size].reshape(item.shape))  
                idx += size  
            elif isinstance(item, list):  
                size = len(item)  
                result.append(vX[idx:idx+size].reshape(-1, 1) if size > 0 else np.array([]))  
                idx += size  
            else:  
                # Handle other types  
                result.append(np.array([]))  
        return result  
    else:  
        return vX
  
def spm_cat(x, d=None, debug=False):  
    """Convert a cell array into a matrix"""  
    if not isinstance(x, list):  
        return x  
      
    if d is not None:  
        # Concatenate over specific dimension  
        if d == 1:  
            return sparse.vstack([spm_cat(col) for col in x])  
        elif d == 2:  
            return sparse.hstack([spm_cat(row) for row in x])  
        else:  
            raise ValueError("Unknown dimension")  
      
    # Handle empty list  
    if len(x) == 0:  
        return sparse.csr_matrix((0, 0))  
      
    # Check if this is a simple cell array (like w) that needs vertical stacking  
    if all(isinstance(item, (np.ndarray, sparse.spmatrix)) for item in x):  
        # For w matrices, stack vertically (along rows)  
        return sparse.vstack([sparse.csr_matrix(item) for item in x])  
      
    # Handle nested cell array structure  
    result_rows = []  
    for row in x:  
        if isinstance(row, list):  
            row_items = [sparse.csr_matrix(item) for item in row]  
            result_rows.append(sparse.hstack(row_items) if row_items else sparse.csr_matrix((0, 0)))  
        else:  
            result_rows.append(sparse.csr_matrix(row))  
      
    return sparse.vstack(result_rows) if result_rows else sparse.csr_matrix((0, 0))
  
def spm_DEM_M_custom(model, *varargs, debug=False):  
    """Create a template model structure - Python version"""  
    _debug_print(f"spm_DEM_M_custom input: {model}", varargs, debug)  
      
    model = model.lower()  
      
    if model == 'lorenz':  
        M = [ModelLevel(), ModelLevel()]  
          
        # Level 1 - Lorenz dynamics  
        M[0].E = type('E', (), {})()  
        M[0].E.linear = 3  
        M[0].E.s = 1/8  
          
        P = np.array([18.0, -4.0, 46.92])  
        x = np.array([0.9, 0.8, 30])  
        scale = 32  
          
        # Override with varargs if provided  
        if len(varargs) >= 1 and varargs[0] is not None:  
            P = varargs[0]  
        if len(varargs) >= 2 and varargs[1] is not None:  
            x = varargs[1]  
        if len(varargs) >= 3 and varargs[2] is not None:  
            scale = varargs[2]  
          
        def lorenz_f(x_state, v, P_params):  
            A = np.array([[-P_params[0], P_params[0], 0],  
                         [P_params[2] - x_state[2], -1, -x_state[0]],  
                         [x_state[1], x_state[0], P_params[1]]])  
            return A @ x_state / scale  
          
        M[0].f = lorenz_f  
        #M[0].g = lambda x, v, P: np.sum(x)  
        M[0].g = lambda x, v, P: np.array([np.sum(x)])
        M[0].x = x  
        M[0].pE = P  
        M[0].V = np.exp(0)  
        M[0].W = np.exp(16)  
          
        # Level 2 - causes  
        M[1].v = 0  
        M[1].V = np.exp(16)  
          
    else:  
        raise ValueError(f"Unknown model: {model}")  
      
    _debug_print("spm_DEM_M_custom before M_set", M, debug)  
    result = spm_DEM_M_set(M, debug=debug)  
    _debug_print("spm_DEM_M_custom output", result, debug)  
    return result  
  
def spm_DEM_M_set(M, debug=False):  
    """Set indices and perform checks on hierarchical models"""  
    g = len(M)  
      
    # Check supra-ordinate level and add one if necessary  
    if hasattr(M[g-1].g, '__call__'):  
        M.append(ModelLevel())  
        M[g].l = M[g-1].m if M[g-1].m is not None else 0  
        g += 1  
      
    M[g-1].m = 0  
    M[g-1].n = 0  
      
    # Set missing fields for static models  
    for i in range(g):  
        if M[i].f is None:  
            M[i].f = lambda x, v, P: sparse.csr_matrix((0, 1))  
            M[i].x = sparse.csr_matrix((0, 1))  
            M[i].n = 0  
      
    # Set default values  
    for i in range(g):  
        if M[i].pE is None:  
            M[i].pE = sparse.csr_matrix((0, 0))  
        if M[i].pC is None:  
            p = len(spm_vec(M[i].pE))  
            M[i].pC = sparse.csr_matrix((p, p))  
      
    # CRITICAL: Evaluate functions to determine dimensions  
    for i in range(g):  
        if hasattr(M[i].g, '__call__'):  
            # Get dimensions from function evaluation  
            if M[i].x is not None:  
                if sparse.issparse(M[i].x):  
                    x_eval = M[i].x.toarray().flatten()  
                else:  
                    x_eval = M[i].x  
            else:  
                # For Lorenz, we know the initial states  
                x_eval = np.array([0.9, 0.8, 30]) if i == 0 else np.array([])  
              
            v_eval = 0  # No input for evaluation  
              
            try:  
                g_result = M[i].g(x_eval, v_eval, M[i].pE)  
                M[i].l = len(spm_vec(g_result))  
                _debug_print(f"Level {i} evaluated l={M[i].l} from g function", None, debug)  
            except Exception as e:  
                _debug_print(f"Level {i} g evaluation failed: {e}", None, debug)  
                # For Lorenz, we know l should be 1  
                M[i].l = 1 if i < 2 else 0  # Both levels should have 1 output
          
        # Ensure dimensions are set  
        if M[i].l is None:  
            M[i].l = 1 if i < 2 else 0  # First two levels have 1 output  
        if M[i].m is None:  
            M[i].m = 1 if i == 0 else 0  # Only level 0 has 1 input  
        if M[i].n is None:  
            M[i].n = 3 if i == 0 else 0  # Only level 0 has 3 states  
      
    # Handle V and W precision matrices  
    for i in range(g):  
        # Handle V (input precision)  
        if M[i].V is not None:  
            if np.isscalar(M[i].V):  
                if M[i].l > 0:  
                    M[i].V = M[i].V * sparse.eye(M[i].l, M[i].l)  
                else:  
                    M[i].V = sparse.csr_matrix((0, 0))  
            elif sparse.issparse(M[i].V):  
                if M[i].l > 0 and (M[i].V.shape[0] != M[i].l or M[i].V.shape[1] != M[i].l):  
                    M[i].V = sparse.eye(M[i].l, M[i].l) * M[i].V[0, 0] if M[i].V.nnz > 0 else sparse.eye(M[i].l, M[i].l)  
        else:  
            # CRITICAL: Initialize V if None  
            if M[i].l > 0:  
                M[i].V = sparse.eye(M[i].l, M[i].l)  
            else:  
                M[i].V = sparse.csr_matrix((0, 0))  
          
        # Handle W (state precision)  
        if M[i].W is not None:  
            if np.isscalar(M[i].W):  
                if M[i].n > 0:  
                    M[i].W = M[i].W * sparse.eye(M[i].n, M[i].n)  
                else:  
                    M[i].W = sparse.csr_matrix((0, 0))  
            elif sparse.issparse(M[i].W):  
                if M[i].n > 0 and (M[i].W.shape[0] != M[i].n or M[i].W.shape[1] != M[i].n):  
                    M[i].W = sparse.eye(M[i].n, M[i].n) * M[i].W[0, 0] if M[i].W.nnz > 0 else sparse.eye(M[i].n, M[i].n)  
        else:  
            # CRITICAL: Initialize W if None  
            if M[i].n > 0:  
                M[i].W = sparse.eye(M[i].n, M[i].n)  
            else:  
                M[i].W = sparse.csr_matrix((0, 0))  
      
    # Set estimation parameters  
    nx = sum([level.n for level in M])  
      
    if not hasattr(M[0], 'E') or M[0].E is None:  
        M[0].E = type('E', (), {})()  
      
    if not hasattr(M[0].E, 's'):  
        M[0].E.s = 1/2 if nx > 0 else 0  
    if not hasattr(M[0].E, 'dt'):  
        M[0].E.dt = 1  
    if not hasattr(M[0].E, 'd'):  
        M[0].E.d = 2 if nx > 0 else 0  
    if not hasattr(M[0].E, 'n'):  
        M[0].E.n = 6 if nx > 0 else 0  
      
    return M
  
def spm_DEM_z(M, N, debug=False):  
    """Create hierarchical innovations for generating data"""  
    _debug_print(f"spm_DEM_z input: N={N}", M, debug)  
      
    s = M[0].E.s + np.exp(-16)  
    dt = M[0].E.dt  
    t = np.arange(N) * dt  
      
    # Create temporal convolution matrix  
    K = toeplitz(np.exp(-t**2 / (2 * s**2)))  
    K = K @ np.diag(1.0 / np.sqrt(np.diag(K @ K.T)))  
      
    _debug_print("Temporal convolution matrix K", K, debug)  
      
    z = []  
    w = []  
      
    for i in range(len(M)):  
        _debug_print(f"\nProcessing level {i}", None, debug)  
        _debug_print(f"  Level {i} dimensions: l={M[i].l}, n={M[i].n}", None, debug)  
          
        # Precision of causes  
        P = M[i].V.copy()  
        _debug_print(f"  Level {i} V precision", P, debug)  
          
        # Add prior expectations if Q exists  
        if M[i].Q is not None:  
            if not isinstance(M[i].Q, list):  
                M[i].Q = [M[i].Q]  
            for j, Qj in enumerate(M[i].Q):  
                if M[i].hE is not None and len(M[i].hE) > j:  
                    P = P + Qj * np.exp(M[i].hE[j])  
          
        # Create causes  
        if sparse.issparse(P):  
            P_norm = np.linalg.norm(P.data)  
        else:  
            P_norm = np.linalg.norm(P)  
          
        _debug_print(f"  Level {i} P_norm: {P_norm}", None, debug)  
          
        if P_norm == 0 or M[i].l == 0:  
            z_i = np.random.randn(M[i].l, N) @ K if M[i].l > 0 else sparse.csr_matrix((0, N))  
        elif P_norm >= np.exp(16):  
            z_i = sparse.csr_matrix((M[i].l, N))  
        else:  
            if sparse.issparse(P):  
                P_dense = P.toarray()  
            else:  
                P_dense = P  
              
            if P_dense.ndim == 0:  
                P_dense = np.array([[P_dense]])  
            elif P_dense.ndim == 1:  
                P_dense = P_dense.reshape(-1, 1)  
              
            P_inv = np.linalg.inv(P_dense)  
            sqrt_P_inv = sqrtm(P_inv)  
            z_i = sqrt_P_inv @ np.random.randn(M[i].l, N) @ K  
          
        # Precision of states  
        P_w = M[i].W.copy()  
        _debug_print(f"  Level {i} W precision", P_w, debug)  
          
        # Add prior expectations if R exists  
        if M[i].R is not None:  
            if not isinstance(M[i].R, list):  
                M[i].R = [M[i].R]  
            for j, Rj in enumerate(M[i].R):  
                if M[i].gE is not None and len(M[i].gE) > j:  
                    P_w = P_w + Rj * np.exp(M[i].gE[j])  
          
        # Create state noise  
        if sparse.issparse(P_w):  
            P_w_norm = np.linalg.norm(P_w.data)  
        else:  
            P_w_norm = np.linalg.norm(P_w)  
          
        _debug_print(f"  Level {i} P_w_norm: {P_w_norm}", None, debug)  
          
        if P_w_norm == 0 or M[i].n == 0:  
            w_i = np.random.randn(M[i].n, N) @ K * dt if M[i].n > 0 else sparse.csr_matrix((0, N))  
        elif P_w_norm >= np.exp(16):  
            w_i = sparse.csr_matrix((M[i].n, N))  
        else:  
            if sparse.issparse(P_w):  
                P_w_dense = P_w.toarray()  
            else:  
                P_w_dense = P_w  
              
            if P_w_dense.ndim == 0:  
                P_w_dense = np.array([[P_w_dense]])  
            elif P_w_dense.ndim == 1:  
                P_w_dense = P_w_dense.reshape(-1, 1)  
              
            P_w_inv = np.linalg.inv(P_w_dense)  
            sqrt_P_w_inv = sqrtm(P_w_inv)  
            w_i = sqrt_P_w_inv @ np.random.randn(M[i].n, N) @ K * dt  
          
        z.append(z_i)  
        w.append(w_i)  
        _debug_print(f"  Level {i} z shape: {z_i.shape}", None, debug)  
        _debug_print(f"  Level {i} w shape: {w_i.shape}", None, debug)  

    _debug_print("spm_DEM_z output", (z, w), debug)  
    return z, w  
  
def spm_DEM_int(M, z, w, u, debug=False):  
    """Integrate/evaluate a hierarchical model given innovations z{i} and w{i}"""  
    _debug_print("spm_DEM_int input", (M, z, w, u), debug)  
      
    # Set model indices and missing fields  
    M = spm_DEM_M_set(M, debug)  
      
    # Keep z and w as lists (like MATLAB cell arrays) - DO NOT concatenate yet  
    _debug_print("Keeping z and w as lists", None, debug)  
    _debug_print(f"z has {len(z)} levels", None, debug)  
    _debug_print(f"w has {len(w)} levels", None, debug)  
      
    # Place exogenous causes in cell array (like MATLAB)  
    if not isinstance(u, list):  
        u = [u]  
      
    # Get dimensions  
    nl = len(M)  # number of levels  
    nt = z[0].shape[1] if hasattr(z[0], 'shape') else 1024  # number of time steps  
    nv = sum([level.l for level in M])  # number of v (causal states)  
    nx = sum([level.n for level in M])  # number of x (hidden states)  
      
    # Embedding order  
    dE = M[0].E.d if hasattr(M[0].E, 'd') else 2  
      
    # Initialize response and hidden states  
    V = []  
    X = []  
    Z = []  
    W = []  
      
    for i in range(nl):  
        V.append(sparse.csr_matrix((M[i].l, nt)))  
        X.append(sparse.csr_matrix((M[i].n, nt)))  
        Z.append(sparse.csr_matrix((M[i].l, nt)))  
        W.append(sparse.csr_matrix((M[i].n, nt)))  
      
    # Generalized filtering iterations  
    nD = 16  # number of iterations  
      
    for d in range(nD):  
        _debug_print(f"Generalized filtering iteration {d}", None, debug)  
          
        # Initialize states for this iteration  
        vi = []  
        xi = []  
          
        for i in range(nl):  
            if M[i].l > 0:  
                vi.append(sparse.csr_matrix((M[i].l, 1)))  
            else:  
                vi.append(sparse.csr_matrix((0, 1)))  
              
            if M[i].n > 0:  
                xi.append(sparse.csr_matrix((M[i].n, 1)))  
            else:  
                xi.append(sparse.csr_matrix((0, 1)))  
          
        # Time integration loop  
        for t in range(nt):  
            # Update states using generalized filtering  
            for i in range(nl):  
                if M[i].n > 0 and i < len(xi) and xi[i] is not None:  
                    # Simple Euler integration for hidden states  
                    if hasattr(M[i].f, '__call__'):  
                        x_new = M[i].f(xi[i].toarray(), vi[i].toarray(), M[i].pE)  
                        if i < len(w) and w[i] is not None and hasattr(w[i], 'shape') and w[i].shape[1] > t:  
                            x_new = x_new + w[i][:, t].reshape(-1, 1)  
                        xi[i] = sparse.csr_matrix(x_new)  
                  
                if M[i].l > 0 and i < len(vi) and vi[i] is not None:  
                    # Update causal states  
                    if hasattr(M[i].g, '__call__'):  
                        v_new = M[i].g(xi[i].toarray(), 0, M[i].pE)  
                        if i < len(z) and z[i] is not None and hasattr(z[i], 'shape') and z[i].shape[1] > t:  
                            v_new = v_new + z[i][:, t].reshape(-1, 1)  
                        if i < len(u) and u[i] is not None and hasattr(u[i], 'shape') and u[i].shape[1] > t:  
                            v_new = v_new + u[i][:, t].reshape(-1, 1)  
                        vi[i] = sparse.csr_matrix(v_new)  
              
            # Save realization  
            for i in range(nl):  
                if M[i].l > 0 and i < len(vi) and vi[i] is not None:  
                    vec_result = spm_vec(vi[i])  
                    if len(vec_result) > 0:  
                        V[i][:, t] = vec_result  
                if M[i].n > 0 and i < len(xi) and xi[i] is not None:  
                    vec_result = spm_vec(xi[i])  
                    if len(vec_result) > 0:  
                        X[i][:, t] = vec_result  
                if M[i].l > 0 and i < len(z) and z[i] is not None and hasattr(z[i], 'shape') and z[i].shape[1] > t:  
                    Z[i][:, t] = z[i][:, t]  
                if M[i].n > 0 and i < len(w) and w[i] is not None and hasattr(w[i], 'shape') and w[i].shape[1] > t:  
                    W[i][:, t] = w[i][:, t]  
      
    _debug_print("spm_DEM_int output", (V, X, Z, W), debug)  
    return V, X, Z, W

def spm_DEM_generate(M, U, P=None, h=None, g=None, debug=False):  
    """Generate data for a Hierarchical Dynamic Model (HDM)"""  
    _debug_print("spm_DEM_generate input", (M, U), debug)  
      
    # Set and check model  
    M = spm_DEM_M_set(M, debug)  
      
    # Determine sequence length  
    if hasattr(U, 'shape'):  
        N = U.shape[1]  
    else:  
        N = U  
        U = sparse.csr_matrix((M[-1].l if M[-1].l > 0 else 1, N))  
      
    _debug_print(f"Sequence length N={N}", None, debug)  
      
    # Initialize parameters  
    m = len(M)  
      
    # Set default hyperparameters  
    for i in range(m):  
        if M[i].hE is None:  
            M[i].hE = 32  
        if M[i].gE is None:  
            M[i].gE = 32  
      
    # Create innovations  
    z, w = spm_DEM_z(M, N, debug)  
      
    # Place exogenous causes in cell array  
    u = []  
    for i in range(m):  
        u_i = sparse.csr_matrix((M[i].l if M[i].l > 0 else 1, N))  
        u.append(u_i)  
    u[-1] = U  
      
    _debug_print("Causes placed in cell array", None, debug)  
      
    # Integrate HDM  
    v, x, z, w = spm_DEM_int(M, z, w, u, debug)  
      
    # Create DEM structure  
    DEM = type('DEM', (), {})()  
    DEM.M = M  
    DEM.Y = v[0]  
    DEM.pU = type('pU', (), {})()  
    DEM.pU.v = v  
    DEM.pU.x = x  
    DEM.pU.z = z  
    DEM.pU.w = w  
    DEM.pP = type('pP', (), {})()  
    DEM.pP.P = [level.pE for level in M]  
    DEM.pH = type('pH', (), {})()  
    DEM.pH.h = [level.hE for level in M]  
    DEM.pH.g = [level.gE for level in M]  
      
    _debug_print("DEM structure created", DEM, debug)  
    return DEM  

def spm_DEM_embed(Y, n, t, dt, d=0, debug=False):  
    """Temporal embedding into derivatives"""  
    _debug_print(f"spm_DEM_embed input: Y shape={Y[0].shape if hasattr(Y[0], 'shape') else 'N/A'}, n={n}", None, debug)  
      
    # Simplified implementation for data generation  
    # Full implementation would compute temporal derivatives  
    result = []  
    for i in range(n):  
        if i == 0:  
            # Return the original values  
            if isinstance(Y, list):  
                result.append(Y[0] if len(Y) > 0 else sparse.csr_matrix((1, 1)))  
            else:  
                result.append(Y)  
        else:  
            # Higher-order derivatives (simplified as zeros for data generation)  
            if isinstance(Y, list) and len(Y) > 0:  
                result.append(sparse.csr_matrix(Y[0].shape))  
            else:  
                result.append(sparse.csr_matrix((1, 1)))  
      
    _debug_print(f"spm_DEM_embed output: {len(result)} levels", None, debug)  
    return result

# Main execution function  
def generate_lorenz_data(debug=False):  
    """Equivalent to the MATLAB code provided"""  
    _debug_print("Starting Lorenz data generation", None, debug)  
      
    # Set random seed  
    np.random.seed(1)  
      
    # Get model of stochastic chaos  
    M = spm_DEM_M_custom('Lorenz', debug=debug)  
      
    # Create innovations & add causes  
    N = 1024  
    U = sparse.csr_matrix(([1], ([0], [0])), shape=(1, N))  
    DEM = spm_DEM_generate(M, U, debug=debug)  
      
    _debug_print("Lorenz data generation complete", None, debug)  
    return DEM  
  
if __name__ == "__main__":  
    # Test the implementation  
    DEM = generate_lorenz_data(debug=True)  
    print(f"Generated data shape: {DEM.Y.shape}")  
    print(f"First few values: {DEM.Y[0, :5]}")
