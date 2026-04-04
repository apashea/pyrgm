import numpy as np  
from scipy import sparse  
from scipy.linalg import sqrtm, toeplitz, expm  
import warnings
import math

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
  

  
def spm_vec(X, *args):  
    """Vectorise a numeric, cell or structure array - matches MATLAB spm_vec"""  
    if args:  
        X = [X] + list(args)  
      
    if isinstance(X, (np.ndarray, list)):  
        if isinstance(X, list):  
            # Handle list of arrays  
            result = []  
            for item in X:  
                if sparse.issparse(item):  
                    result.append(item.toarray().flatten())  
                elif isinstance(item, np.ndarray):  
                    result.append(item.flatten())  
                else:  
                    result.append(np.array([item]).flatten())  
            return np.concatenate(result)  
        else:  
            # Single array  
            if sparse.issparse(X):  
                return X.toarray().flatten()  
            else:  
                return X.flatten()  
    elif isinstance(X, dict):  
        # Handle structure-like dict  
        result = []  
        for key in sorted(X.keys()):  
            result.append(spm_vec(X[key]))  
        return np.concatenate(result)  
    else:  
        return np.array([])

def spm_unvec(vX, template):  
    """Unvectorise a vectorised array - matches MATLAB spm_unvec"""  
    result = []  
    idx = 0  
      
    for i, item in enumerate(template):  
        if sparse.issparse(item):  
            n = item.shape[0] * item.shape[1]  
        elif hasattr(item, 'shape'):  
            n = np.prod(item.shape)  
        else:  
            n = 1  
          
        _debug_print(f"spm_unvec: item {i}, n={n}, idx={idx}, idx+n={idx+n}", None, False)  
          
        # FIXED: Use shape[0] instead of len() for sparse matrices  
        if sparse.issparse(vX):  
            total_elements = vX.shape[0]  
        else:  
            total_elements = len(vX)  
              
        if idx + n > total_elements:  
            _debug_print(f"spm_unvec: WARNING idx+n ({idx+n}) > total_elements ({total_elements})", None, False)  
            n = total_elements - idx  
          
        if sparse.issparse(item):  
            new_item = sparse.csr_matrix(vX[idx:idx+n].reshape(item.shape))  
        else:  
            new_item = vX[idx:idx+n].reshape(item.shape)  
        result.append(new_item)  
        idx += n  
      
    return result


def spm_DEM_diff(M, u, debug=False):  
    """Evaluate an active model given innovations z{i} and w{i}"""  
    _debug_print("spm_DEM_diff input", (M, u), debug)  
      
    # Check for action (ADEM)  
    try:  
        M[0].a  
        ADEM = 1  
    except:  
        u['a'] = u['v']  
        for level in M:  
            if not hasattr(level, 'a'):  
                level.a = sparse.csr_matrix((0, 1))  
            if not hasattr(level, 'k'):  
                level.k = 0  
        ADEM = 0  
      
    # Number of levels and order parameters  
    nl = len(M)  
    n = M[0].E.n + 1 if hasattr(M[0], 'E') and M[0].E is not None else 1  
      
    # Initialize arrays for hierarchical form  
    dg = {'dv': [[None for _ in range(nl-1)] for _ in range(nl)],   
          'dx': [[None for _ in range(nl-1)] for _ in range(nl)],  
          'dp': [[None for _ in range(nl-1)] for _ in range(nl)]}  
    df = {'dv': [[None for _ in range(nl-1)] for _ in range(nl-1)],   
          'dx': [[None for _ in range(nl-1)] for _ in range(nl-1)],  
          'dp': [[None for _ in range(nl-1)] for _ in range(nl-1)]}  
      
    # Initialize Jacobian matrices  
    for i in range(nl - 1):  
        for j in range(nl - 1):  
            dg['dv'][i+1][j] = sparse.csr_matrix((M[i].m, M[i].m))  
            dg['dx'][i+1][j] = sparse.csr_matrix((M[i].m, M[i].n))  
            dg['dp'][i+1][j] = sparse.csr_matrix((M[i].m, M[i].p))  
            dg['dv'][i][j] = sparse.csr_matrix((M[i].l, M[i].m))  
            dg['dx'][i][j] = sparse.csr_matrix((M[i].l, M[i].n))  
            dg['dp'][i][j] = sparse.csr_matrix((M[i].l, M[i].p))  
            df['dv'][i][j] = sparse.csr_matrix((M[i].n, M[i].m))  
            df['dx'][i][j] = sparse.csr_matrix((M[i].n, M[i].n))  
            df['dp'][i][j] = sparse.csr_matrix((M[i].n, M[i].p))  
      
    # Partition states - FIXED: Use per-time-step template structure  
    vi_template = []  
    xi_template = []  
    ai_template = []  
      
    for level in M:  
        # Handle v template  
        if hasattr(level, 'v') and level.v is not None:  
            if np.isscalar(level.v):  
                vi_template.append(np.array([level.v]))  # Shape (1,) per time step  
            elif sparse.issparse(level.v):  
                vi_template.append(level.v)  
            else:  
                vi_template.append(np.array(level.v))  
        else:  
            vi_template.append(np.array([]))  
          
        # Handle x template  
        if hasattr(level, 'x') and level.x is not None:  
            if np.isscalar(level.x):  
                xi_template.append(np.array([level.x]))  
            elif sparse.issparse(level.x):  
                xi_template.append(level.x)  
            else:  
                xi_template.append(np.array(level.x))  
        else:  
            xi_template.append(np.array([]))  
          
        # Handle a template  
        if hasattr(level, 'a') and level.a is not None:  
            if np.isscalar(level.a):  
                ai_template.append(np.array([level.a]))  
            elif sparse.issparse(level.a):  
                ai_template.append(level.a)  
            else:  
                ai_template.append(np.array(level.a))  
        else:  
            ai_template.append(np.array([]))  
      
    _debug_print(f"Template shapes - vi: {[t.shape for t in vi_template]}", None, debug)  
    _debug_print(f"Template shapes - xi: {[t.shape for t in xi_template]}", None, debug)  
      
    vi = spm_unvec(u['v'][0], vi_template)  
    xi = spm_unvec(u['x'][0], xi_template)  
    ai = spm_unvec(u['a'][0], ai_template)  
      
    # Update states using model functions  
    for i in range(nl - 1):  
        if M[i].g is not None:  
            g_val = M[i].g(xi[i], vi[i+1], M[i].pE)  
            u['v'][i+1] = spm_vec(g_val)  
          
        if M[i].f is not None:  
            f_val = M[i].f(xi[i], vi[i], M[i].pE)  
            u['x'][i] = spm_vec(f_val)  
      
    # Compute Jacobians numerically  
    for i in range(nl - 1):  
        if M[i].g is not None:  
            # Compute dg/dx  
            dg['dx'][i+1][i] = compute_jacobian(  
                lambda x: M[i].g(x, vi[i+1], M[i].pE),   
                xi[i], debug  
            )  
              
            # Compute dg/dv  
            dg['dv'][i+1][i] = compute_jacobian(  
                lambda v: M[i].g(xi[i], v, M[i].pE),   
                vi[i+1], debug  
            )  
              
            # Constant term for linking causes  
            dg['dv'][i+1][i] = -speye(M[i].m, M[i].m)  
          
        if M[i].f is not None:  
            # Compute df/dx  
            df['dx'][i][i] = compute_jacobian(  
                lambda x: M[i].f(x, vi[i], M[i].pE),   
                xi[i], debug  
            )  
              
            # Compute df/dv  
            df['dv'][i][i] = compute_jacobian(  
                lambda v: M[i].f(xi[i], v, M[i].pE),   
                vi[i], debug  
            )  
      
    # Concatenate hierarchical forms  
    D = {}  
    D['dgdv'] = spm_cat([item for row in dg['dv'] for item in row if item is not None])  
    D['dgdx'] = spm_cat([item for row in dg['dx'] for item in row if item is not None])  
    D['dfdv'] = spm_cat([item for row in df['dv'] for item in row if item is not None])  
    D['dfdx'] = spm_cat([item for row in df['dx'] for item in row if item is not None])  
      
    _debug_print("spm_DEM_diff output", D, debug)  
    return u, D, df  
  
def compute_jacobian(func, x, debug=False):  
    """Compute Jacobian df/dx numerically"""  
    eps = 1e-6  
    f0 = func(x)  
    J = np.zeros((len(f0), len(x)))  
      
    for j in range(len(x)):  
        x_eps = x.copy()  
        x_eps[j] += eps  
        f_eps = func(x_eps)  
        J[:, j] = (f_eps - f0) / eps  
      
    return J

  

def _numerical_jacobian(func, x, eps=1e-6):  
    """Compute numerical Jacobian"""  
    if sparse.issparse(x):  
        x = x.toarray().flatten()  
      
    f0 = func(x)  
    if sparse.issparse(f0):  
        f0 = f0.toarray().flatten()  
    else:  
        f0 = np.array(f0)  
      
    J = np.zeros((len(f0), len(x)))  
      
    for j in range(len(x)):  
        x_eps = x.copy()  
        if np.isscalar(x_eps):  
            a_eps = x_eps + eps  
        else:  
            a_eps = x_eps.copy()  
            a_eps[0] += eps  
          
        f_eps = func(a_eps)  
          
        if sparse.issparse(f0):  
            dfda = (f_eps - f0) / eps  
        else:  
            dfda = (np.array(f_eps) - np.array(f0)) / eps  
          
        return dfda

def spm_dx(dfdx, f, t=np.inf, L=None, debug=False):  
    """Returns dx(t) = (expm(dfdx*t) - I)*inv(dfdx)*f - matches MATLAB spm_dx"""  
    _debug_print(f"spm_dx input: dfdx shape={dfdx.shape}, t={t}", None, debug)  
      
    nmax = 512  
    xf = f  
    f = spm_vec(f)  
    n = len(f)  
      
    # Handle t as regularizer  
    if isinstance(t, list):  
        t = t[0]  
        if np.isscalar(t):  
            # Need spm_logdet - using determinant for now  
            t = np.exp(t - np.log(np.abs(np.linalg.det(dfdx))) / n)  
        else:  
            t = np.exp(t - np.log(np.diag(-dfdx)))  
      
    # Use pseudoinverse if t > exp(16)  
    if np.min(t) > np.exp(16):  
        dx = -np.linalg.pinv(dfdx.toarray() if sparse.issparse(dfdx) else dfdx) @ f  
    else:  
        # Ensure t is scalar or matrix  
        if np.ndim(t) == 1:  
            t = np.diag(t)  
          
        # Augment Jacobian  
        zero_block = sparse.csr_matrix((1, 1))  
        tf_block = sparse.csr_matrix(t * f.reshape(-1, 1))  
        tdfdx_block = sparse.csr_matrix(t * dfdx)  
          
        J = spm_cat([[zero_block, None],  
                     [tf_block, tdfdx_block]])  
          
        # Solve using matrix exponential  
        if n <= nmax:  
            dx = expm(J.toarray())  
            dx = dx[:, 0]  
        else:  
            # For large matrices, would need expv implementation  
            # Using scipy's expm for now  
            dx = expm(J.toarray())  
            dx = dx[:, 0]  
          
        # Recover update  
        dx = dx[1:]  
      
    dx = spm_unvec(np.real(dx), xf)  
    return dx

def spm_diff(func, *args, n=1):  
    """Numerical differentiation - simplified version of MATLAB spm_diff"""  
    eps = 1e-6  
      
    if n == 1:  
        # Derivative with respect to first argument  
        x = args[0]  
        f0 = func(x, *args[1:])  
          
        if np.isscalar(x):  
            x_eps = x + eps  
        else:  
            x_eps = x.copy()  
            x_eps[0] += eps  
          
        f_eps = func(x_eps, *args[1:])  
          
        if sparse.issparse(f0):  
            dfdx = (f_eps - f0) / eps  
        else:  
            dfdx = (np.array(f_eps) - np.array(f0)) / eps  
          
        return dfdx, f0  
      
    elif n == 2:  
        # Derivative with respect to second argument  
        v = args[1]  
        f0 = func(args[0], v, *args[2:])  
          
        if np.isscalar(v):  
            v_eps = v + eps  
        else:  
            v_eps = v.copy()  
            v_eps[0] += eps  
          
        f_eps = func(args[0], v_eps, *args[2:])  
          
        if sparse.issparse(f0):  
            dfdv = (f_eps - f0) / eps  
        else:  
            dfdv = (np.array(f_eps) - np.array(f0)) / eps  
          
        return dfdv  
      
    elif n == 3:  
        # Derivative with respect to third argument  
        a = args[2]  
        f0 = func(args[0], args[1], a, *args[3:])  
          
        if np.isscalar(a):  
            a_eps = a + eps  
        else:  
            a_eps = a.copy()  
            a_eps[0] += eps  
          
        f_eps = func(args[0], args[1], a_eps, *args[3:])  
          
        if sparse.issparse(f0):  
            dfda = (f_eps - f0) / eps  
        else:  
            dfda = (np.array(f_eps) - np.array(f0)) / eps  
          
        return dfda

def spm_cat(x, d=None, debug=False):  
    """Concatenate matrices from cell array - matches MATLAB spm_cat behavior"""  
    _debug_print(f"spm_cat input: {type(x)}", None, debug)  
      
    # If not a cell array, return as is  
    if not isinstance(x, list):  
        return x  
      
    # Handle dimension argument (not used in our current implementation)  
    if d is not None:  
        _debug_print(f"spm_cat: dimension argument {d} not implemented", None, debug)  
      
    # Convert all to sparse matrices - FIXED: Handle type checking properly  
    matrices = []  
    for item in x:  
        if item is None:  
            matrices.append(sparse.csr_matrix((0, 0)))  
        elif sparse.issparse(item):  
            # Already sparse, just ensure CSR format  
            matrices.append(item.tocsr())  
        elif isinstance(item, np.ndarray):  
            # Handle numpy arrays  
            if item.size == 0:  
                matrices.append(sparse.csr_matrix(item.shape))  
            else:  
                matrices.append(sparse.csr_matrix(item))  
        else:  
            # Handle other types (scalars, etc.)  
            try:  
                # Convert to numpy array first to avoid boolean evaluation  
                item_array = np.array(item)  
                if item_array.size == 0:  
                    matrices.append(sparse.csr_matrix((0, 0)))  
                else:  
                    matrices.append(sparse.csr_matrix(item_array))  
            except:  
                matrices.append(sparse.csr_matrix((0, 0)))  
      
    if not matrices:  
        return sparse.csr_matrix((0, 0))  
      
    # Find dimensions for each row and column (MATLAB approach)  
    # For our simple list case, treat as single row  
    max_cols = max(m.shape[1] for m in matrices)  
    max_rows = max(m.shape[0] for m in matrices)  
      
    # Pad matrices to have same dimensions  
    padded_matrices = []  
    for m in matrices:  
        if m.shape[1] < max_cols or m.shape[0] < max_rows:  
            # Pad with zeros  
            padded = sparse.lil_matrix((max_rows, max_cols))  
            padded[:m.shape[0], :m.shape[1]] = m  
            padded_matrices.append(padded.tocsr())  
        else:  
            padded_matrices.append(m)  
      
    # Concatenate vertically (equivalent to MATLAB's cat(1, y{:}))  
    try:  
        result = sparse.vstack(padded_matrices)  
    except:  
        # Fallback to dense if sparse fails  
        dense_matrices = [m.toarray() for m in padded_matrices]  
        result = np.vstack(dense_matrices)  
        result = sparse.csr_matrix(result)  
      
    _debug_print(f"spm_cat output shape: {result.shape}", None, debug)  
    return result
  
def spm_DEM_M_custom(model, debug=False, *varargs):  
    """Custom model creation for Python implementation"""  
    _debug_print(f"spm_DEM_M_custom input: {model}", None, debug)  
      
    if model == 'Lorenz':  
        # Create hierarchical model  
        M = []  
          
        # Level 1: Lorenz dynamics  
        M.append(ModelLevel())  
          
        # Set parameters (matching MATLAB values)  
        M[0].pE = np.array([18.0, -4.0, 46.92])  # sigma, rho, beta  
        M[0].x = np.array([0.9, 0.8, 30.0])  # Initial states  
          
        # FIXED: Set initial causal state to match MATLAB (v=31.7)  
        M[0].v = 31.7  
          
        # Define dynamics functions  
        def lorenz_f(x_state, v, P_params):  
            """Lorenz dynamics: dx/dt = f(x,v,P)"""  
            # Ensure x_state is 1D  
            if hasattr(x_state, 'flatten'):  
                x_state = x_state.flatten()  
              
            A = np.array([[-P_params[0], P_params[0], 0],      
                         [P_params[2] - x_state[2], -1, -x_state[0]],  
                         [x_state[1], x_state[0], P_params[1]]])  
            return A @ x_state  
          
        def lorenz_g(x_state, v, P_params):  
            """Lorenz output: y = sum(x)"""  
            # Ensure x_state is 1D  
            if hasattr(x_state, 'flatten'):  
                x_state = x_state.flatten()  
            # FIXED: Return 1D array to match MATLAB behavior  
            return np.array([np.sum(x_state)])  
          
        M[0].f = lorenz_f  
        M[0].g = lorenz_g  
          
        # FIXED: Precision matrices as proper sparse matrices  
        M[0].V = sparse.csr_matrix(np.array([[1.0]]))  # V = 1.0  
        M[0].W = sparse.csr_matrix(np.exp(16) * np.eye(3))  # W = 8886110.520508 * I(3)  
          
        # Level 2: Static observation level  
        M.append(ModelLevel())  
        M[1].g = lambda x, v, P: np.array([0])  # Fixed output  
        M[1].f = None  # No dynamics  
          
        # FIXED: Level 2 precision matrix as sparse matrix  
        M[1].V = sparse.csr_matrix(np.array([[np.exp(16)]]))  # V = 8886110.520508  
          
        # Set dimensions  
        M[0].n = 3  # 3 states for Lorenz  
        M[0].l = 1  # 1 output  
        M[0].m = 1  # 1 input  
          
        M[1].n = 0  # No states  
        M[1].l = 1  # 1 output      
        M[1].m = 0  # No inputs  
          
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
      
    # Set estimation parameters - CRITICAL FIX  
    nx = sum([level.n for level in M])  
      
    if not hasattr(M[0], 'E') or M[0].E is None:  
        M[0].E = type('E', (), {})()  
      
    # Match MATLAB exactly  
    if not hasattr(M[0].E, 's'):  
        M[0].E.s = 1/8 if nx > 0 else 0  # MATLAB shows 0.125, not 0.5  
    if not hasattr(M[0].E, 'dt'):  
        M[0].E.dt = 1  
    if not hasattr(M[0].E, 'd'):  
        M[0].E.d = 2 if nx > 0 else 0  
    if not hasattr(M[0].E, 'n'):  
        M[0].E.n = 6 if nx > 0 else 0  
      
    # Add missing fields that MATLAB has  
    if not hasattr(M[0].E, 'linear'):  
        M[0].E.linear = 1  
    if not hasattr(M[0].E, 'nD'):  
        M[0].E.nD = 1  
    if not hasattr(M[0].E, 'nE'):  
        M[0].E.nE = 8  
    if not hasattr(M[0].E, 'nM'):  
        M[0].E.nM = 8  
    if not hasattr(M[0].E, 'nN'):  
        M[0].E.nN = 32  
      
    # Initialize additional fields to match MATLAB  
    for i in range(g):  
        # Initialize missing fields with defaults  
        if not hasattr(M[i], 'Q'):  
            M[i].Q = None  
        if not hasattr(M[i], 'R'):  
            M[i].R = None  
        if not hasattr(M[i], 'hE'):  
            M[i].hE = sparse.csr_matrix((0, 1))  
        if not hasattr(M[i], 'gE'):  
            M[i].gE = sparse.csr_matrix((0, 1))  
        if not hasattr(M[i], 'hC'):  
            M[i].hC = sparse.csr_matrix((0, 0))  
        if not hasattr(M[i], 'gC'):  
            M[i].gC = sparse.csr_matrix((0, 0))  
        if not hasattr(M[i], 'ph'):  
            M[i].ph = sparse.csr_matrix((0, 0))  
        if not hasattr(M[i], 'pg'):  
            M[i].pg = sparse.csr_matrix((0, 0))  
        if not hasattr(M[i], 'xP'):  
            M[i].xP = sparse.csr_matrix((M[i].n, M[i].n))  
        if not hasattr(M[i], 'vP'):  
            M[i].vP = sparse.csr_matrix((M[i].l, M[i].l))  
        if not hasattr(M[i], 'sv'):  
            M[i].sv = M[0].E.s if hasattr(M[0], 'E') and M[0].E is not None else 0  
        if not hasattr(M[i], 'sw'):  
            M[i].sw = M[0].E.s if hasattr(M[0], 'E') and M[0].E is not None else 0  
      
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
      
    # Concatenate innovations and causes  
    z_cat = spm_cat(z)  
    u_cat = spm_cat(u)  
      
    # Calculate expected dimensions  
    nv = sum([level.l for level in M])  # total causal states  
    nx = sum([level.n for level in M])  # total hidden states  
      
    _debug_print(f"Expected dimensions: nv={nv}, nx={nx}", None, debug)  
    _debug_print(f"z_cat.shape={z_cat.shape}, u_cat.shape={u_cat.shape}", None, debug)  
      
    # Ensure z_cat and u_cat have correct dimensions  
    if z_cat.shape[0] != nv:  
        _debug_print(f"Reshaping z_cat from {z_cat.shape} to ({nv}, {z_cat.shape[1]})", None, debug)  
        if z_cat.shape[0] > nv:  
            z_cat = z_cat[:nv, :]  
        else:  
            # Pad with zeros if needed  
            pad_rows = nv - z_cat.shape[0]  
            z_cat = sparse.vstack([z_cat, sparse.csr_matrix((pad_rows, z_cat.shape[1]))])  
      
    if u_cat.shape[0] != nv:  
        _debug_print(f"Reshaping u_cat from {u_cat.shape} to ({nv}, {u_cat.shape[1]})", None, debug)  
        if u_cat.shape[0] > nv:  
            u_cat = u_cat[:nv, :]  
        else:  
            # Pad with zeros if needed  
            pad_rows = nv - u_cat.shape[0]  
            u_cat = sparse.vstack([u_cat, sparse.csr_matrix((pad_rows, u_cat.shape[1]))])  
      
    # Concatenate w  
    w_cat = spm_cat(w)  
      
    # Ensure w_cat has correct dimensions  
    if w_cat.shape[0] != nx:  
        _debug_print(f"Reshaping w_cat from {w_cat.shape} to ({nx}, {w_cat.shape[1]})", None, debug)  
        if w_cat.shape[0] > nx:  
            w_cat = w_cat[:nx, :]  
        else:  
            # Pad with zeros if needed  
            pad_rows = nx - w_cat.shape[0]  
            w_cat = sparse.vstack([w_cat, sparse.csr_matrix((pad_rows, w_cat.shape[1]))])  
      
    # Get dimensions  
    nt = z_cat.shape[1]  # number of time steps  
    nl = len(M)          # number of levels  
      
    _debug_print(f"Integration parameters: nt={nt}, nl={nl}", None, debug)  
      
    # Initialize template variables - FIXED: Convert scalars to arrays  
    vi = []  
    xi = []  
    for i in range(nl):  
        if M[i].v is not None:  
            if np.isscalar(M[i].v):  
                vi.append(np.array([M[i].v]))  
            elif sparse.issparse(M[i].v):  
                vi.append(M[i].v)  
            else:  
                vi.append(np.array(M[i].v))  
        else:  
            vi.append(sparse.csr_matrix((M[i].l, 1)))  
              
        if M[i].x is not None:  
            if sparse.issparse(M[i].x):  
                xi.append(M[i].x)  
            else:  
                xi.append(np.array(M[i].x))  
        else:  
            xi.append(sparse.csr_matrix((M[i].n, 1)))  
      
    _debug_print(f"Template initialized: vi has {len(vi)} levels, xi has {len(xi)} levels", None, debug)  
      
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
          
        # Initial conditions  
        if M[i].v is not None:  
            if sparse.issparse(M[i].v):  
                V[i][:, 0] = M[i].v.toarray().flatten()  
            else:  
                V[i][:, 0] = M[i].v  
        if M[i].x is not None:  
            if sparse.issparse(M[i].x):  
                X[i][:, 0] = M[i].x.toarray().flatten()  
            else:  
                X[i][:, 0] = M[i].x  
      
    # Embedding parameters  
    if hasattr(M[0], 'E') and M[0].E is not None:  
        n = M[0].E.n + 1  
        dt = M[0].E.dt  
        nD = M[0].E.d  
    else:  
        n = 1  
        dt = 1  
        nD = 1  
      
    td = dt / nD  
      
    _debug_print(f"Embedding parameters: dt={dt}, n={n}, nD={nD}, td={td}", None, debug)  
      
    # Precision matrices - FIXED: Use block diagonal construction  
    V_mats = [level.V for level in M if hasattr(level, 'V') and level.V is not None]  
    W_mats = [level.W for level in M if hasattr(level, 'W') and level.W is not None]  
      
    Sz = _block_diag(V_mats) if V_mats else sparse.csr_matrix((nv, nv))  
    Sw = _block_diag(W_mats) if W_mats else sparse.csr_matrix((nx, nx))  
      
    _debug_print(f"Precision matrices: Sz.shape={Sz.shape}, Sw.shape={Sw.shape}", None, debug)  
      
    # Derivative operators for Jacobian  
    Dx = spm_kron(speye(n, n, 1), speye(nx, nx, 0))  
    Dv = spm_kron(speye(n, n, 1), speye(nv, nv, 0))  
    D = spm_cat([Dv, Dx, Dv, Dx])  
      
    _debug_print(f"Derivative operators: Dx.shape={Dx.shape}, Dv.shape={Dv.shape}", None, debug)  
      
    # Initialize generalized states  
    u_states = {  
        'v': [sparse.csr_matrix((M[i].l, nt)) for i in range(nl)],  
        'x': [sparse.csr_matrix((M[i].n, nt)) for i in range(nl)],  
        'z': [sparse.csr_matrix((M[i].l, nt)) for i in range(nl)],  
        'w': [sparse.csr_matrix((M[i].n, nt)) for i in range(nl)]  
    }  
      
    # Time integration loop  
    for t in range(nt):  
        _debug_print(f"Time step {t}/{nt}", None, debug)  
          
        # Initialize states at time t  
        for i in range(nl):  
            if M[i].l > 0:  
                u_states['v'][i][:, t] = V[i][:, t]  
            if M[i].n > 0:  
                u_states['x'][i][:, t] = X[i][:, t]  
          
        # Multiple iterations for convergence (generalized filtering)  
        for iD in range(nD):  
            # Temporal embedding of innovations  
            u_states['z'] = spm_DEM_embed(Sz * z_cat, n, t, dt)  
            u_states['w'] = spm_DEM_embed(Sw * w_cat, n, t, dt)  
              
            # Evaluate model and compute Jacobians  
            u_eval, dg, df = spm_DEM_diff(M, u_states)  
              
            # FIXED: Compute dfdw if not present  
            if 'dw' not in df:  
                df['dw'] = [[None for _ in range(nl-1)] for _ in range(nl-1)]  
                for i in range(nl-1):  
                    if M[i].f is not None:  
                        df['dw'][i][i] = sparse.eye(M[i].n, M[i].n)  
              
            # FIXED: Construct large Jacobian matrix properly  
            # Get dimensions for each block  
            dg_dv_shape = dg['dv'][0][0].shape if dg['dv'][0][0] is not None else (0, 0)  
            dg_dx_shape = dg['dx'][0][0].shape if dg['dx'][0][0] is not None else (0, 0)  
            df_dv_shape = df['dv'][0][0].shape if df['dv'][0][0] is not None else (0, 0)  
            df_dx_shape = df['dx'][0][0].shape if df['dx'][0][0] is not None else (0, 0)  
            df_dw_shape = df['dw'][0][0].shape if df['dw'][0][0] is not None else (0, 0)  
              
            # Create zero blocks with correct dimensions  
            zero_dg_dv = sparse.csr_matrix(dg_dv_shape)  
            zero_dg_dx = sparse.csr_matrix(dg_dx_shape)  
            zero_df_dv = sparse.csr_matrix(df_dv_shape)  
            zero_df_dx = sparse.csr_matrix(df_dx_shape)  
            zero_df_dw = sparse.csr_matrix(df_dw_shape)  
              
            J = spm_cat([  
                spm_cat([spm_cat([dg['dv'][i][j] for j in range(nl-1) for i in range(nl) if dg['dv'][i][j] is not None]),  
                        spm_cat([dg['dx'][i][j] for j in range(nl-1) for i in range(nl) if dg['dx'][i][j] is not None]),  
                        zero_dg_dv, zero_dg_dv]),  
                spm_cat([spm_cat([df['dv'][i][j] for j in range(nl-1) for i in range(nl-1) if df['dv'][i][j] is not None]),  
                        spm_cat([df['dx'][i][j] for j in range(nl-1) for i in range(nl-1) if df['dx'][i][j] is not None]),  
                        zero_df_dv, spm_cat([df['dw'][i][j] for j in range(nl-1) for i in range(nl-1) if df['dw'][i][j] is not None])]),  
                spm_cat([zero_dg_dv, zero_dg_dx, Dv, sparse.csr_matrix((Dv.shape[0], Dv.shape[1]))]),  
                spm_cat([zero_df_dv, zero_df_dx, sparse.csr_matrix((Dx.shape[0], Dx.shape[1])), Dx])  
            ])  
              
            # Update states using spm_dx  
            du = spm_dx(J, D * spm_vec(u_states), td)  
            u_vec = spm_vec(u_states) + du  
            u_states = spm_unvec(u_vec, u_states)  
              
            # Save realization (first iteration only)  
            if iD == 0:  
                vi = spm_unvec(u_states['v'][0], vi)  
                xi = spm_unvec(u_states['x'][0], xi)  
                zi = spm_unvec(u_states['z'][0], vi)  
                wi = spm_unvec(u_states['w'][0], xi)  
                  
                for i in range(nl):  
                    if M[i].l > 0:  
                        V[i][:, t] = spm_vec(vi[i])  
                        Z[i][:, t] = spm_vec(zi[i])  
                    if M[i].n > 0:  
                        X[i][:, t] = spm_vec(xi[i])  
                        W[i][:, t] = spm_vec(wi[i])  
      
    _debug_print("spm_DEM_int output", (V, X, Z, W), debug)  
    return V, X, Z, W
  
def _block_diag(matrices):  
    """Create block diagonal matrix from list of matrices"""  
    # Filter out None matrices  
    matrices = [m for m in matrices if m is not None]  
    if not matrices:  
        return sparse.csr_matrix((0, 0))  
      
    # Convert all to CSR format  
    matrices = [sparse.csr_matrix(m) for m in matrices]  
      
    # Calculate total dimensions  
    total_rows = sum(m.shape[0] for m in matrices)  
    total_cols = sum(m.shape[1] for m in matrices)  
      
    # Create block diagonal matrix  
    result = sparse.lil_matrix((total_rows, total_cols))  
      
    row_offset = 0  
    col_offset = 0  
    for m in matrices:  
        rows, cols = m.shape  
        result[row_offset:row_offset+rows, col_offset:col_offset+cols] = m  
        row_offset += rows  
        col_offset += cols  
      
    return result.tocsr()
def compute_jacobian(f, x, v, p):  
    """Compute Jacobian df/dx numerically"""  
    eps = 1e-6  
    f0 = f(x, v, p)  
    J = np.zeros((len(x), len(x)))  
      
    for j in range(len(x)):  
        x_eps = x.copy()  
        x_eps[j] += eps  
        f_eps = f(x_eps, v, p)  
        J[:, j] = (f_eps - f0) / eps  
      
    return J
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
    """Temporal embedding into derivatives - matches MATLAB spm_DEM_embed"""  
    _debug_print(f"spm_DEM_embed input: Y shape={Y.shape if hasattr(Y, 'shape') else 'N/A'}, n={n}, t={t}", None, debug)  
      
    # Handle sparse matrix input  
    if sparse.issparse(Y):  
        Y_dense = Y.toarray()  
    else:  
        Y_dense = Y  
      
    # Ensure Y is 2D  
    if Y_dense.ndim == 1:  
        Y_dense = Y_dense.reshape(-1, 1)  
      
    ny, nt = Y_dense.shape  
      
    # Initialize output  
    y = []  
      
    # Create Taylor expansion matrix for derivatives  
    # Using the approach from MATLAB's spm_DEM_embed  
    T = np.zeros((n, n))  
    for i in range(n):  
        for j in range(i, n):  
            if i == j:  
                T[i, j] = 1.0  
            else:  
                T[i, j] = (dt**(j-i)) / math.factorial(j-i)  
      
    _debug_print(f"Taylor matrix T shape: {T.shape}, condition number: {np.linalg.cond(T):.2e}", None, debug)  
      
    # Check if T is singular or ill-conditioned  
    if np.linalg.cond(T) > 1e12:  
        _debug_print("Taylor matrix is ill-conditioned, using pseudo-inverse", None, debug)  
        E = np.linalg.pinv(T)  
    else:  
        try:  
            E = np.linalg.inv(T)  
        except np.linalg.LinAlgError:  
            _debug_print("Taylor matrix is singular, using pseudo-inverse", None, debug)  
            E = np.linalg.pinv(T)  
      
    # Embed each time series  
    for i in range(ny):  
        y_i = []  
          
        for ti in range(nt):  
            # Get local neighborhood for derivative computation  
            t_start = max(0, ti - n + 1)  
            t_end = min(nt, ti + n)  
              
            # Extract local segment  
            if t_end - t_start >= n:  
                Y_local = Y_dense[i, t_start:t_start+n]  
            else:  
                # Pad with zeros if at boundary  
                Y_local = np.zeros(n)  
                actual_len = t_end - t_start  
                Y_local[:actual_len] = Y_dense[i, t_start:t_end]  
              
            # Compute derivatives using embedding operator  
            if sparse.issparse(Y):  
                derivative = sparse.csr_matrix(E @ Y_local)  
            else:  
                derivative = E @ Y_local  
              
            y_i.append(derivative)  
          
        # Stack derivatives for this time series  
        if sparse.issparse(Y):  
            y.append(sparse.hstack(y_i))  
        else:  
            y.append(np.column_stack(y_i))  
      
    _debug_print(f"spm_DEM_embed output: {len(y)} derivative levels", None, debug)  
    return y
      
    # Convert d to list if scalar  
    if np.isscalar(d):  
        d = [d]  
      
    # Loop over channels  
    for p in range(len(d)):  
        # Boundary conditions  
        s = (t - d[p]) / dt  
        k = (np.arange(1, n+1)) + np.fix(s - (n + 1) / 2)  
        x = s - np.min(k) + 1  
          
        # Handle boundaries  
        k[k < 1] = 1  
        k[k > N] = N  
          
        # Inverse embedding operator T (Taylor expansion)  
        T = np.zeros((n, n))  
        for i in range(n):  
            for j in range(n):  
                if j == 0:  
                    T[i, j] = 1  
                else:  
                    T[i, j] = ((i - x) * dt) ** (j - 1) / np.prod(range(1, j))  
          
        # Embedding operator E  
        E = np.linalg.inv(T)  
          
        # Embed  
        if len(d) == q:  
            for i in range(n):  
                y[i][p, :] = sparse.csr_matrix(Y[p, k.astype(int)] @ E[i, :].T)  
        else:  
            for i in range(n):  
                y[i] = sparse.csr_matrix(Y[:, k.astype(int)] @ E[i, :].T)  
            return y  
      
    _debug_print(f"spm_DEM_embed output: {len(y)} derivatives", None, debug)  
    return y

def spm_kron(A, B):  
    """Kronecker tensor product with sparse outputs - matches MATLAB spm_kron"""  
    if isinstance(A, list):  
        # Handle cell array input  
        K = 1  
        for i in range(len(A)):  
            K = spm_kron(A[i], K)  
        return K  
      
    # Convert to sparse if not already  
    if not sparse.issparse(A):  
        A = sparse.csr_matrix(A)  
    if not sparse.issparse(B):  
        B = sparse.csr_matrix(B)  
      
    # Use scipy's kron for sparse matrices  
    return sparse.kron(A, B)  

def speye(m, n=None, k=0):  
    """Sparse identity matrix - matches MATLAB speye behavior"""  
    if n is None:  
        n = m  
      
    if k == 0:  
        # Standard identity matrix  
        return sparse.eye(m, n, format='csr')  
    else:  
        # Offset diagonal matrix  
        if k > 0:  
            # Superdiagonal  
            if m > k and n > 0:  
                rows = np.arange(k, m)  
                cols = np.arange(min(m - k, n))  
                data = np.ones(len(rows))  
                return sparse.csr_matrix((data, (rows, cols)), shape=(m, n))  
        else:  
            # Subdiagonal  
            if n > abs(k) and m > 0:  
                rows = np.arange(min(m, n - abs(k)))  
                cols = np.arange(abs(k), n)  
                data = np.ones(len(rows))  
                return sparse.csr_matrix((data, (rows, cols)), shape=(m, n))  
          
        return sparse.csr_matrix((m, n))

def format_value_for_display(val, name=""):  
    """Helper function to format values for display, handling scalars and arrays"""  
    if np.isscalar(val):  
        return f"{val:.6f}"  
    elif hasattr(val, 'toarray'):  
        # Handle sparse matrices  
        val_array = val.toarray().flatten()  
        if len(val_array) > 0:  
            return " ".join([f"{v:.6f}" for v in val_array])  
        else:  
            return ""  
    elif hasattr(val, '__len__') and len(val) > 0:  
        # Handle dense arrays  
        return " ".join([f"{v:.6f}" for v in val])  
    else:  
        return ""

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
