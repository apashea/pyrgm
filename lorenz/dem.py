import numpy as np  
from scipy import sparse  
from scipy.linalg import sqrtm, toeplitz  
import warnings  
  
def spm_vec(X):  
    """Vectorise a numeric, cell or structure array"""  
    if isinstance(X, np.ndarray):  
        return X.ravel()  
    elif isinstance(X, list):  
        result = []  
        for item in X:  
            result.extend(spm_vec(item))  
        return np.array(result)  
    else:  
        return np.array([])  
  
def spm_unvec(vX, X):  
    """Unvectorise back to original shape"""  
    if isinstance(X, np.ndarray):  
        return vX.reshape(X.shape)  
    else:  
        return vX  
  
class ModelLevel:  
    """Represents a single level in the hierarchical model"""  
    def __init__(self):  
        self.g = None  # y(t) = g(x,v,P)  
        self.f = None  # dx/dt = f(x,v,P)  
        self.pE = None  # prior expectation of parameters  
        self.pC = None  # prior covariances of parameters  
        self.hE = None  # prior expectation of h (cause noise)  
        self.hC = None  # prior covariances of h  
        self.gE = None  # prior expectation of g (state noise)  
        self.gC = None  # prior covariances of g  
        self.Q = None   # precision components (input noise)  
        self.R = None   # precision components (state noise)  
        self.V = None   # fixed precision (input noise)  
        self.W = None   # fixed precision (state noise)  
        self.m = None   # number of inputs v(i+1)  
        self.n = None   # number of states x(i)  
        self.l = None   # number of outputs v(i)  
        self.x = None   # hidden states  
        self.v = None   # causal states  
        self.E = None   # estimation parameters  
  
def spm_DEM_M_custom(model, *varargs):  
    """Create a template model structure - Python version"""  
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
          
        # Define dynamics function  
        def lorenz_f(x_state, v, P_params):  
            A = np.array([[-P_params[0], P_params[0], 0],  
                         [P_params[2] - x_state[2], -1, -x_state[0]],  
                         [x_state[1], x_state[0], P_params[1]]])  
            return A @ x_state / scale  
          
        M[0].f = lorenz_f  
        M[0].g = lambda x, v, P: np.sum(x)  
        M[0].x = x  
        M[0].pE = P  
        M[0].V = np.exp(0)  # precision of observation noise  
        M[0].W = np.exp(16)  # precision of state noise  
          
        # Level 2 - causes  
        M[1].v = 0  
        M[1].V = np.exp(16)  
          
    else:  
        raise ValueError(f"Unknown model: {model}")  
      
    return spm_DEM_M_set(M)  
  
def spm_DEM_M_set(M):  
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
          
        # Check and set V and W  
        if M[i].V is None:  
            if M[i].l is not None and M[i].l > 0:  
                M[i].V = sparse.eye(M[i].l)  
            else:  
                M[i].V = sparse.csr_matrix((0, 0))  
          
        if M[i].W is None:  
            if M[i].n is not None and M[i].n > 0:  
                M[i].W = sparse.eye(M[i].n)  
            else:  
                M[i].W = sparse.csr_matrix((0, 0))  
          
        # Convert to sparse matrices if needed  
        if isinstance(M[i].V, np.ndarray):  
            M[i].V = sparse.csr_matrix(M[i].V)  
        if isinstance(M[i].W, np.ndarray):  
            M[i].W = sparse.csr_matrix(M[i].W)  
      
    # Set estimation parameters  
    nx = sum(spm_vec([level.n for level in M]))  
      
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
      
    # Check functions and set dimensions  
    for i in range(g-1, -1, -1):  
        if M[i].x is None and M[i].n is not None and M[i].n > 0:  
            M[i].x = sparse.csr_matrix((M[i].n, 1))  
          
        # Evaluate g to get dimensions  
        if hasattr(M[i].g, '__call__'):  
            x_eval = M[i].x.toarray().flatten() if sparse.issparse(M[i].x) else M[i].x  
            v_eval = M[i].v if M[i].v is not None else 0  
            p_eval = M[i].pE  
              
            try:  
                g_result = M[i].g(x_eval, v_eval, p_eval)  
                M[i].l = len(spm_vec(g_result))  
                M[i].n = len(spm_vec(x_eval))  
                M[i].m = M[i].l  # For this simple case  
            except:  
                pass  
      
    return M  
  
def spm_DEM_z(M, N):  
    """Create hierarchical innovations for generating data"""  
    s = M[0].E.s + np.exp(-16)  
    dt = M[0].E.dt  
    t = np.arange(N) * dt  
      
    # Create temporal convolution matrix  
    K = toeplitz(np.exp(-t**2 / (2 * s**2)))  
    K = K @ np.diag(1.0 / np.sqrt(np.diag(K @ K.T)))  
      
    z = []  
    w = []  
      
    for i in range(len(M)):  
        # Precision of causes  
        P = M[i].V.copy()  
          
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
          
        if P_norm == 0:  
            z_i = np.random.randn(M[i].l, N) @ K  
        elif P_norm >= np.exp(16):  
            z_i = sparse.csr_matrix((M[i].l, N))  
        else:  
            P_inv = np.linalg.inv(P.toarray()) if sparse.issparse(P) else np.linalg.inv(P)  
            sqrt_P_inv = sqrtm(P_inv)  
            z_i = sqrt_P_inv @ np.random.randn(M[i].l, N) @ K  
          
        # Precision of states  
        P_w = M[i].W.copy()  
          
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
          
        if P_w_norm == 0:  
            w_i = np.random.randn(M[i].n, N) @ K  
        elif P_w_norm >= np.exp(16):  
            w_i = sparse.csr_matrix((M[i].n, N))  
        else:  
            P_w_inv = np.linalg.inv(P_w.toarray()) if sparse.issparse(P_w) else np.linalg.inv(P_w)  
            sqrt_P_w_inv = sqrtm(P_w_inv)  
            w_i = sqrt_P_w_inv @ np.random.randn(M[i].n, N) @ K  
          
        z.append(z_i)  
        w.append(w_i)  
      
    return z, w  
  
def spm_DEM_int(M, z, w, u):  
    """Integrate HDM to obtain causal (v) and hidden states (x)"""  
    # Simple Euler integration for demonstration  
    # In practice, this would use more sophisticated integration  
    m = len(M)  
    N = z[0].shape[1]  
      
    v = []  
    x = []  
      
    for i in range(m):  
        v_i = np.zeros((M[i].l, N))  
        x_i = np.zeros((M[i].n, N))  
          
        # Initial conditions  
        if M[i].v is not None:  
            v_i[:, 0] = M[i].v  
        if M[i].x is not None:  
            if sparse.issparse(M[i].x):  
                x_i[:, 0] = M[i].x.toarray().flatten()  
            else:  
                x_i[:, 0] = M[i].x  
          
        # Simple integration  
        for t in range(1, N):  
            # Update states  
            if hasattr(M[i].f, '__call__') and M[i].n > 0:  
                dx = M[i].f(x_i[:, t-1], v_i[:, t-1], M[i].pE)  
                x_i[:, t] = x_i[:, t-1] + dx * M[0].E.dt + w[i][:, t]  
              
            # Update outputs/causes  
            if hasattr(M[i].g, '__call__'):  
                v_i[:, t] = M[i].g(x_i[:, t], u[i][:, t] if i < len(u) else 0, M[i].pE) + z[i][:, t]  
          
        v.append(v_i)  
        x.append(x_i)  
      
    return v, x, z, w  
  
def spm_DEM_generate(M, U, P=None, h=None, g=None):  
    """Generate data for a Hierarchical Dynamic Model (HDM)"""  
    # Set and check model  
    M = spm_DEM_M_set(M)  
      
    # Determine sequence length  
    if hasattr(U, 'shape'):  
        N = U.shape[1]  
    else:  
        N = U  
        U = sparse.csr_matrix((M[-1].l, N))  
      
    # Initialize parameters  
    m = len(M)  
      
    # Set default hyperparameters  
    for i in range(m):  
        if M[i].hE is None:  
            M[i].hE = 32  
        if M[i].gE is None:  
            M[i].gE = 32  
      
    # Create innovations  
    z, w = spm_DEM_z(M, N)  
      
    # Place exogenous causes in cell array  
    u = []  
    for i in range(m):  
        u_i = sparse.csr_matrix((M[i].l, N))  
        u.append(u_i)  
    u[-1] = U  
      
    # Integrate HDM  
    v, x, z, w = spm_DEM_int(M, z, w, u)  
      
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
      
    return DEM  
  
# Main execution function  
def generate_lorenz_data():  
    """Equivalent to the MATLAB code provided"""  
    # Set random seed  
    np.random.seed(1)  
      
    # Get model of stochastic chaos  
    M = spm_DEM_M_custom('Lorenz')  
      
    # Create innovations & add causes  
    N = 1024  
    U = sparse.csr_matrix(([1], ([0], [0])), shape=(1, N))  
    DEM = spm_DEM_generate(M, U)  
      
    return DEM  
  
if __name__ == "__main__":  
    # Test the implementation  
    DEM = generate_lorenz_data()  
    print(f"Generated data shape: {DEM.Y.shape}")  
    print(f"First few values: {DEM.Y[0, :5]}")
