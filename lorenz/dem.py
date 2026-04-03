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
    _debug_print("spm_unvec input", (vX, X), debug)  
      
    if isinstance(X, np.ndarray):  
        result = vX.reshape(X.shape)  
    elif isinstance(X, list):  
        result = []  
        idx = 0  
        for item in X:  
            if hasattr(item, 'shape'):  
                size = np.prod(item.shape)  
            else:  
                size = len(item) if item else 0  
            result.append(vX[idx:idx+size].reshape(item.shape if hasattr(item, 'shape') else (len(item),)))  
            idx += size  
    else:  
        result = vX  
      
    _debug_print("spm_unvec output", result, debug)  
    return result  
  
def spm_cat(x, d=None, debug=False):  
    """Convert a cell array into a matrix"""  
    _debug_print(f"spm_cat input (d={d})", x, debug)  
      
    if not isinstance(x, list):  
        _debug_print("spm_cat output (not list)", x, debug)  
        return x  
      
    if d is not None:  
        if d == 1:  
            result = sparse.vstack([spm_cat(col, debug=debug) for col in x])  
        elif d == 2:  
            result = sparse.hstack([spm_cat(row, debug=debug) for row in x])  
        else:  
            raise ValueError("Unknown dimension")  
        _debug_print(f"spm_cat output (dim={d})", result, debug)  
        return result  
      
    if len(x) == 0:  
        result = sparse.csr_matrix((0, 0))  
        _debug_print("spm_cat output (empty list)", result, debug)  
        return result  
      
    # Find dimensions  
    max_rows = 0  
    max_cols = 0  
    for row in x:  
        for item in row:  
            if hasattr(item, 'shape') and item.shape[0] > 0:  
                max_rows = max(max_rows, item.shape[0])  
            if hasattr(item, 'shape') and item.shape[1] > 0:  
                max_cols = max(max_cols, item.shape[1])  
      
    _debug_print(f"spm_cat found max dimensions", (max_rows, max_cols), debug)  
      
    if max_rows == 0 or max_cols == 0:  
        result = sparse.csr_matrix((0, 0))  
        _debug_print("spm_cat output (no valid dimensions)", result, debug)  
        return result  
      
    # Fill with sparse matrices  
    result_rows = []  
    for i, row in enumerate(x):  
        row_items = []  
        for j, item in enumerate(row):  
            if sparse.issparse(item) and item.nnz > 0:  
                row_items.append(item)  
            elif hasattr(item, 'shape') and item.shape[0] > 0:  
                row_items.append(sparse.csr_matrix(item))  
            else:  
                row_items.append(sparse.csr_matrix((max_rows, max_cols)))  
          
        if row_items:  
            result_row = sparse.hstack(row_items)  
            result_rows.append(result_row)  
            _debug_print(f"spm_cat row {i} stacked", result_row, debug)  
      
    if result_rows:  
        result = sparse.vstack(result_rows)  
    else:  
        result = sparse.csr_matrix((0, 0))  
      
    _debug_print("spm_cat final output", result, debug)  
    return result  
  
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
        M[0].g = lambda x, v, P: np.sum(x)  
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
    _debug_print("spm_DEM_M_set input", M, debug)  
      
    g = len(M)  
      
    # Check supra-ordinate level and add one if necessary  
    if hasattr(M[g-1].g, '__call__'):  
        M.append(ModelLevel())  
        M[g].l = M[g-1].m if M[g-1].m is not None else 0  
        g += 1  
        _debug_print("Added supra-ordinate level", g, debug)  
      
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
            p = len(spm_vec(M[i].pE, debug))  
            M[i].pC = sparse.csr_matrix((p, p))  
      
    # Ensure dimensions are set  
    for i in range(g):  
        if M[i].l is None:  
            M[i].l = 0  
        if M[i].m is None:  
            M[i].m = 0  
        if M[i].n is None:  
            M[i].n = 0  
      
    _debug_print("Dimensions after initialization", [(i, M[i].l, M[i].m, M[i].n) for i in range(g)], debug)  
      
    # Handle V and W precision matrices  
    for i in range(g):  
        _debug_print(f"Level {i} V before processing", M[i].V, debug)  
        _debug_print(f"Level {i} W before processing", M[i].W, debug)  
          
        # Handle V  
        if M[i].V is not None:  
            if np.isscalar(M[i].V):  
                if M[i].l > 0:  
                    M[i].V = M[i].V * sparse.eye(M[i].l, M[i].l)  
                else:  
                    M[i].V = sparse.csr_matrix((0, 0))  
        else:  
            if M[i].l > 0:  
                M[i].V = sparse.eye(M[i].l, M[i].l)  
            else:  
                M[i].V = sparse.csr_matrix((0, 0))  
          
        # Handle W  
        if M[i].W is not None:  
            if np.isscalar(M[i].W):  
                if M[i].n > 0:  
                    M[i].W = M[i].W * sparse.eye(M[i].n, M[i].n)  
                else:  
                    M[i].W = sparse.csr_matrix((0, 0))  
        else:  
            if M[i].n > 0:  
                M[i].W = sparse.eye(M[i].n, M[i].n)  
            else:  
                M[i].W = sparse.csr_matrix((0, 0))  
          
        _debug_print(f"Level {i} V after processing", M[i].V, debug)  
        _debug_print(f"Level {i} W after processing", M[i].W, debug)  
      
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
      
    _debug_print("spm_DEM_M_set output", M, debug)  
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
      
    # Innovations - MATLAB concatenates z and u  
    _debug_print("Concatenating innovations", None, debug)  
    z_cat = spm_cat(z, debug=debug)  
    u_cat = spm_cat(u, debug=debug)  
    _debug_print(f"z_cat shape: {z_cat.shape}", z_cat, debug)  
    _debug_print(f"u_cat shape: {u_cat.shape}", u_cat, debug)  
      
    z = z_cat + u_cat  
    w = spm_cat(w, debug=debug)  
    _debug_print(f"Combined z shape: {z.shape}", z, debug)  
    _debug_print(f"Combined w shape: {w.shape}", w, debug)  
      
    # Number of states and parameters  
    nt = z.shape[1]  # number of time steps  
    nl = len(M)  # number of levels  
    nv = sum(spm_vec([level.l for level in M]))  # number of v (causal states)  
    nx = sum(spm_vec([level.n for level in M]))  # number of x (hidden states)  
      
    _debug_print(f"Dimensions: nt={nt}, nl={nl}, nv={nv}, nx={nx}", None, debug)  
      
    # Order parameters (n=1 for static models)  
    dt = M[0].E.dt  # time step  
    n = M[0].E.n + 1 if hasattr(M[0].E, 'n') else 2  # order of embedding  
    nD = M[0].E.nD if hasattr(M[0].E, 'nD') else 1  # number of iterations per sample  
    td = dt / nD  # integration time for D-Step  
      
    _debug_print(f"Integration params: dt={dt}, n={n}, nD={nD}, td={td}", None, debug)  
      
    # Initialize cell arrays for derivatives  
    u_states = type('u', (), {})()  
    u_states.v = [sparse.csr_matrix((nv, 1)) for _ in range(n)]  
    u_states.x = [sparse.csr_matrix((nx, 1)) for _ in range(n)]  
    u_states.z = [sparse.csr_matrix((nv, 1)) for _ in range(n)]  
    u_states.w = [sparse.csr_matrix((nx, 1)) for _ in range(n)]  
      
    # Hyperparameters  
    ph = type('ph', (), {})()  
    ph.h = [level.hE for level in M]  
    ph.g = [level.gE for level in M]  
      
    # Initialize with starting conditions  
    vi = [level.v for level in M]  
    xi = [level.x for level in M]  
    u_states.v[0] = sparse.csr_matrix(spm_vec(vi)).reshape(-1, 1)  
    u_states.x[0] = sparse.csr_matrix(spm_vec(xi)).reshape(-1, 1)  
      
    _debug_print("Initial states set", None, debug)  
      
    # Derivatives for Jacobian of D-step  
    Dx = sparse.kron(sparse.eye(n, n, 1), sparse.eye(nx, nx, 0))  
    Dv = sparse.kron(sparse.eye(n, n, 1), sparse.eye(nv, nv, 0))  
    D = spm_cat([[Dv, Dx], [Dv, Dx]], debug=debug)  
    dfdw = sparse.kron(sparse.eye(n, n), sparse.eye(nx, nx))  
      
    # Initialize conditional estimators of states to be saved (V and X)  
    V = []  
    X = []  
    Z = []  
    W = []  
      
    for i in range(nl):  
        V.append(sparse.csr_matrix((M[i].l, nt)))  
        X.append(sparse.csr_matrix((M[i].n, nt)))  
        Z.append(sparse.csr_matrix((M[i].l, nt)))  
        W.append(sparse.csr_matrix((M[i].n, nt)))  
      
    _debug_print("State arrays initialized", None, debug)  
      
    # Iterate over sequence (t) and within for static models  
    for t in range(nt):  
        for iD in range(nD):  
            # Get generalised motion of random fluctuations  
            # Sampling time  
            ts = (t + (iD) / nD) * dt  
              
            # Derivatives of innovations (and exogenous input)  
            u_states.z = spm_DEM_embed(z, n, ts, dt, debug=debug)  
            u_states.w = spm_DEM_embed(w, n, ts, dt, debug=debug)  
              
            # Simplified evaluation (full implementation would use spm_DEM_diff)  
            for i in range(n):  
                if i < n - 1:  
                    u_states.v[i+1] = u_states.v[i]  # Simplified derivative  
                    u_states.x[i+1] = u_states.x[i]  # Simplified derivative  
              
            # Save realization  
            vi = spm_unvec(u_states.v[0].toarray().flatten(), vi, debug)  
            xi = spm_unvec(u_states.x[0].toarray().flatten(), xi, debug)  
            zi = spm_unvec(u_states.z[0].toarray().flatten(), vi, debug)  
            wi = spm_unvec(u_states.w[0].toarray().flatten(), xi, debug)  
              
            if iD == 0:  
                for i in range(nl):  
                    if M[i].l > 0:  
                        V[i][:, t] = spm_vec(vi[i])  
                    if M[i].n > 0:  
                        X[i][:, t] = spm_vec(xi[i])  
                    if M[i].l > 0:  
                        Z[i][:, t] = spm_vec(zi[i])  
                    if M[i].n > 0:  
                        W[i][:, t] = spm_vec(wi[i])  
              
            # Break for static models  
            if nt == 1:  
                break  
              
            # Simplified Jacobian update  
            # Full implementation would construct proper Jacobian  
            du = sparse.csr_matrix((4 * (nv + nx), 1))  # Simplified  
              
            # Unpack and update  
            vec_u = spm_cat([u_states.v[0], u_states.x[0], u_states.z[0], u_states.w[0]], debug=debug)  
            vec_u = vec_u + du  
      
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
