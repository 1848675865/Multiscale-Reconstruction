import numpy as np
import math
import igraph as ig
import pickle
import numpy as np
from scipy.optimize import fsolve

N=100
M=10 
load_file = 'WS_network.pkl'
with open(load_file, 'rb') as f:
    loaded_data = pickle.load(f)
G = loaded_data['G']  
simplicial_matrix = loaded_data['node_sim2_list']  
set_subpopulation = loaded_data['set_subpopulation']  
sub_adj_matrix = loaded_data['adj']  
sub_G = loaded_data['sub_G']  
sim2 = G.cliques(3,3) 

def X_Y_i(state_adj_set,t):
    X_U_t = []
    Y_U_t = []
    X_A_t = []
    Y_A_t = []

    for state_adj in range(t-1):
        xu = []
        yu = []
        xa = []
        ya = []
        for node in range(N):
            xiut = (state_adj_set[state_adj][0][node]+state_adj_set[state_adj][2][node])*(state_adj_set[state_adj+1][0][node]+state_adj_set[state_adj+1][1][node])*(state_adj_set[state_adj+1][1][node]+state_adj_set[state_adj+1][3][node])
            xu.append(xiut)
            yiut = (state_adj_set[state_adj][0][node]+state_adj_set[state_adj][2][node])*(state_adj_set[state_adj+1][0][node]+state_adj_set[state_adj+1][1][node])*(state_adj_set[state_adj+1][0][node]+state_adj_set[state_adj+1][2][node])
            yu.append(yiut)
            xiat = (state_adj_set[state_adj][0][node]+state_adj_set[state_adj][2][node])*(state_adj_set[state_adj+1][2][node]+state_adj_set[state_adj+1][3][node])*(state_adj_set[state_adj+1][1][node]+state_adj_set[state_adj+1][3][node])
            xa.append(xiat)
            yiat = (state_adj_set[state_adj][0][node]+state_adj_set[state_adj][2][node])*(state_adj_set[state_adj+1][2][node]+state_adj_set[state_adj+1][3][node])*(state_adj_set[state_adj+1][0][node]+state_adj_set[state_adj+1][2][node])
            ya.append(yiat)
            
        X_U_t.append(xu)
        Y_U_t.append(yu)
        X_A_t.append(xa)
        Y_A_t.append(ya)
    return X_U_t, Y_U_t, X_A_t, Y_A_t

def claculate_theta(state_adj_set,t):
    THETA = []
    for t in range(t-1):
        theta_time = []
        for node in range(N):
            theta = np.sum(state_adj_set[t][1]+state_adj_set[t][3])-(state_adj_set[t][1][node]+state_adj_set[t][3][node])
            theta_time.append(theta)
        THETA.append(theta_time)
    return THETA

def claculate_gamma(x_all, theta_all, y_all, initial_guess=0.9999):  
    def gamma_equation(gamma, x_i, y_i, theta_i):
        try:
            gamma = float(gamma)
            if gamma <= 1e-10 or gamma >= 1 - 1e-10:
                return 1e10
            lhs = 0
            rhs = sum(y * theta for y, theta in zip(y_i, theta_i))

            for x, theta in zip(x_i, theta_i):
                if theta == 0:
                    continue
                gamma_term = gamma**(theta / M)

                if not np.isfinite(gamma_term):
                    return 1e10
                denominator = 1 - gamma_term
                if abs(denominator) < 1e-10:
                    return 1e10
                term = x * theta * gamma_term / denominator
                if not np.isfinite(term):
                    return 1e10
                lhs += term
            result = lhs - rhs
            return result if np.isfinite(result) else 1e10
        except (ZeroDivisionError, OverflowError, FloatingPointError, ValueError):
            return 1e10
    gamma = [] 

    for i in range(N): 
        x_i = [row[i] for row in x_all]
        theta_i = [row[i] for row in theta_all]
        y_i = [row[i] for row in y_all]

        gamma_sol = fsolve(lambda g: gamma_equation(g, x_i, y_i, theta_i),
                           x0=initial_guess, maxfev=2000)
        print(f"Node {i}: gamma = {gamma_sol}")
        gamma.append(gamma_sol)

    return gamma

def claculate_F(gamma_u,THETA):
    F_U = []
    for time in range(t-1):
        F_U_time = []
        for node in range(N):
            f_u = ((gamma_u[node][0]**(THETA[time][node]/M))/(1-(gamma_u[node][0]**(THETA[time][node]/M))))-((gamma_u[node][0]**(THETA[time][node]/M))*math.log(gamma_u[node][0]**(THETA[time][node]/M))/((1-(gamma_u[node][0]**(THETA[time][node]/M)))**2))           
            F_U_time.append(f_u)
        F_U.append(F_U_time)
    return F_U

def claculate_G(gamma_u,THETA):
    G_U = []
    for time in range(t-1):
        G_U_time = []
        for node in range(N):
            g_u = (gamma_u[node][0]**(THETA[time][node]/M))/((1-(gamma_u[node][0]**(THETA[time][node]/M)))**2)
            G_U_time.append(g_u)
        G_U.append(G_U_time)
    return G_U

def claculate_FEI(G_U,X_U_t):
    FEI_U = []
    for time in range(t-1):
        FEI_U_time = []
        for node in range(N):
            fei_u = X_U_t[time][node]*G_U[time][node]
            FEI_U_time.append(fei_u)
        FEI_U.append(FEI_U_time)
    return FEI_U

def claculate_TAO(F_U,Y_U_t,X_U_t):
    TAO_U = []
    for time in range(t-1):
        TAO_U_time = []
        for node in range(N):
            tao_u = Y_U_t[time][node]-X_U_t[time][node]*F_U[time][node]
            TAO_U_time.append(tao_u)
        TAO_U.append(TAO_U_time)
    return TAO_U

def claculate_I(state_adj_set):
    I = []
    for time in range(t-1):
        I_SUB = []
        for sub in range(M):
            a = 0
            for node in set_subpopulation[sub]:
                a+= state_adj_set[time][1][node]+state_adj_set[time][3][node]
            I_SUB.append(a)
        I.append(I_SUB)
    return I

def A1(node,FEI_list,I):
    A1=np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            a=0
            for time in range(t-1):
                a+= FEI_list[time][node]*I[time][i]*I[time][j]
            A1[i][j]=a
    return A1

def A2(TAO_list,I,node):
    A2 = np.zeros((M,1))
    for sub in range(M):
        a = 0
        for time in range(t-1):
            a += TAO_list[time][node]*I[time][sub]
        A2[sub] = a
    return A2

def claculate_RMN(node,FEI_U,TAO_U,I):
    A_left_U = A1(node,FEI_U,I)
    A_right_U = A2(TAO_U,I,node)
    x_U,_,_,_= np.linalg.lstsq(A_left_U, A_right_U, rcond=None)
    return x_U

load_file = 'WS传播.pkl'
with open(load_file, 'rb') as f:
    loaded_data = pickle.load(f)
state_adj_set = loaded_data['state_adj'][200:]

t=500
X_U_t, Y_U_t= X_Y_i(state_adj_set,t)
THETA = claculate_theta(state_adj_set,t)
gamma_u = claculate_gamma(X_U_t, THETA, Y_U_t, initial_guess=0.7)
F_U = claculate_F(gamma_u,THETA)
G_U = claculate_G(gamma_u,THETA)
FEI_U = claculate_FEI(G_U,X_U_t)
TAO_U = claculate_TAO(F_U,Y_U_t,X_U_t)
I = claculate_I(state_adj_set)

RUMN_LIST = []
for node in range(N):
    RU_mn= claculate_RMN(node,FEI_U,TAO_U,I)
    RUMN_LIST.append(RU_mn)
print(RUMN_LIST)

def clc_AUROC_AUPR(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    thresholds = np.unique(scores)[::-1]
    tprs, fprs, precisions = [], [], []
    
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
    
    tprs = np.concatenate(([0], tprs, [1]))
    fprs = np.concatenate(([0], fprs, [1]))
    precisions = np.concatenate(([1.0], precisions, [precisions[-1]]))
    recalls = tprs.copy()
    
    AUROC = np.trapz(tprs, fprs)
    AUPR = np.trapz(precisions, recalls)
    
    return AUROC, AUPR

def evaluate_network(w_scores, w_true):
    scores = w_scores
    labels = w_true
    
    auroc, aupr = clc_AUROC_AUPR(scores, labels)
    
    tp = np.sum((scores == 1) & (labels == 1))
    fp = np.sum((scores == 1) & (labels == 0))
    fn = int(np.sum((scores == 0) & (labels == 1)))
    tn = int(np.sum((scores == 0) & (labels == 0)))
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    tpr = tp / P if P > 0 else 0.0
    fpr = fp / N if N > 0 else 0.0
    
    sr = np.sqrt(tpr * (1 - fpr))
    
    return auroc,aupr,sr

def get_average_value_trimmed(RUMN_LIST_big):
    average_value = []
    for sub in range(M):
        selected_arrays = [RUMN_LIST_big[i] for i in set_subpopulation[sub]]
        stacked = np.stack(selected_arrays, axis=0)
        mean_array = np.mean(stacked, axis=0)
        average_value.append(mean_array)
    
    return average_value

def count_continuous_nonzero(yy, zero_idx, tolerance=0):
    sep = len(yy)
    count = 0
    total_freq = 0
    gap = 0
    for i in range(zero_idx, sep):
        if yy[i] > 0:
            count += 1
            total_freq += yy[i]
            gap = 0
        else:
            gap += 1
            if gap > tolerance:
                break
    gap = 0
    for i in range(zero_idx - 1, -1, -1):
        if yy[i] > 0:
            count += 1
            total_freq += yy[i]
            gap = 0
        else:
            gap += 1
            if gap > tolerance:
                break
    return count, total_freq

def find_best_sep(y, sep_range):
    ymin, ymax = np.min(y), np.max(y)
    best_sep = sep_range[0]
    best_score = (-1, -1)
    best_yy = None
    best_edges = None

    for sep in sep_range:
        yy, edges = np.histogram(y, bins=sep, range=(ymin, ymax))
        zero_idx = np.searchsorted(edges, 0.0, side='right') - 1
        zero_idx = np.clip(zero_idx, 0, sep - 1)

        count, total_freq = count_continuous_nonzero(yy, zero_idx, tolerance=0)

        coverage = count / sep
        score = (coverage * math.sqrt(sep), total_freq)

        if score > best_score:
            best_score = score
            best_sep = sep
            best_yy = yy
            best_edges = edges

    return best_sep, best_yy, best_edges

def find_threshold(yy, edges, sep):
    ymin = edges[0]
    ymax = edges[-1]
    x = np.linspace(ymin, ymax, sep)
    half_win = math.ceil(sep * 0.2)

    sepxy = []
    for sep1 in range(sep):
        left  = max(sep1 - half_win, 0)
        right = min(sep1 + half_win, sep - 1)
        neighbors = yy[left:right + 1]
        count = np.sum(neighbors - yy[sep1] >= 0)
        sepxy.append(count)

    sepxy = np.array(sepxy)
    sepid = np.where(sepxy == np.max(sepxy))[0]
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    valid = [idx for idx in sepid if bin_centers[idx] > 0]
    if not valid:
        return None

    threshold = x[valid[0]]
    return threshold


def estsr_single_cut(dtrs, diag_idx=None, sep_range=None):
    dtrs = np.array(dtrs).flatten()
    m = len(dtrs)
    xadj = np.ones(m)

    if diag_idx is not None:
        mask = np.ones(m, dtype=bool)
        mask[diag_idx] = False
        y = dtrs[mask]          
        orig_indices = np.where(mask)[0]
    else:
        y = dtrs
        orig_indices = np.arange(m)

    if sep_range is None:
        sep_range = range(3, 10)

    best_sep, best_yy, best_edges = find_best_sep(y, sep_range)
    threshold = find_threshold(best_yy, best_edges, best_sep)
    if threshold is None:
        return xadj  # 全为1，即保留所有边
    sub_index = np.where(y <= threshold)[0]
    orig_zero_index = orig_indices[sub_index]
    xadj[orig_zero_index] = 0

    if diag_idx is not None:
        xadj[diag_idx] = 0

    return xadj

def process_array_list(array_list, sep_range=None):
    results = []
    for i, arr in enumerate(array_list):
        xadj = estsr_single_cut(arr, diag_idx=i, sep_range=sep_range)
        results.append(xadj)
    return np.array(results)

lamda = 0.4
log_factor = np.log(1 - lamda)
RUMN_LIST = [arr / log_factor for arr in RUMN_LIST]
average_value = get_average_value_trimmed(RUMN_LIST)
a = process_array_list(average_value, sep_range=None)
auroc, aupr, sr = evaluate_network(a, sub_adj_matrix)