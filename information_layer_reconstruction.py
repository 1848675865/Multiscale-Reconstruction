import numpy as np
from scipy.optimize import fsolve
import pickle
import math

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

def claculate_X_Y(state_adj_set):
    X_t = []
    Y_t = []
    for time in range(t-1):
        x_t = []
        y_t = []
        for node in range(N):
            Xi = (state_adj_set[time][0][node]+state_adj_set[time][1][node])*(state_adj_set[time+1][2][node]+state_adj_set[time+1][3][node])
            Yi = (state_adj_set[time][0][node]+state_adj_set[time][1][node])*(state_adj_set[time+1][0][node]+state_adj_set[time+1][1][node])
            x_t.append(Xi)
            y_t.append(Yi)
        X_t.append(x_t)
        Y_t.append(y_t)
    return X_t,Y_t

def claculate_theta(state_adj_set):
    THETA_t = []
    for time in range(t-1):
        theta_t = []
        for node in range(N):
            theta = np.sum(state_adj_set[time][2]+state_adj_set[time][3])-(state_adj_set[time][2][node]+state_adj_set[time][3][node])
            theta_t.append(theta)
        THETA_t.append(theta_t)
    return THETA_t

def claculate_gamma(X_t, THETA_t, Y_t, initial_guess=0.9999):  
    def gamma_equation_improved(gamma, x_i, y_i, theta_i):
        gamma = float(gamma)
        if gamma <= 1e-10 or gamma >= 1 - 1e-10:
            return 1e10
        lhs = 0
        rhs = sum(y * theta for y, theta in zip(y_i, theta_i))
        for x, theta in zip(x_i, theta_i):
            if theta == 0:
                continue
            gamma_theta = gamma**theta
            denominator = 1 - gamma_theta
            if abs(denominator) < 1e-10:
                return 1e10
            term = x * theta * gamma_theta / denominator
            lhs += term    
        result = lhs - rhs
        return result if np.isfinite(result) else 1e10
    
    gamma = [] 
    
    for i in range(N):
        x_i = [row[i] for row in X_t]
        theta_i = [row[i] for row in THETA_t]
        y_i = [row[i] for row in Y_t]
        
        gamma_sol = fsolve(lambda g: gamma_equation_improved(g, x_i, y_i, theta_i), 
                          x0=initial_guess, maxfev=2000)
        
        gamma.append(gamma_sol)
    
    return gamma

def claculate_F_G(gamma,THETA_t):
    FT = []
    GT = []
    for time in range(t-1):
        ft = []
        gt = []
        for node in range(N):
            fi = (gamma[node][0]**THETA_t[time][node]/(1-gamma[node][0]**THETA_t[time][node]))-(gamma[node][0]**THETA_t[time][node]/(1-gamma[node][0]**THETA_t[time][node])**2)*math.log(gamma[node][0]**THETA_t[time][node])
            gi = gamma[node][0]**THETA_t[time][node]/((1-gamma[node][0]**THETA_t[time][node])**2)
            ft.append(fi)
            gt.append(gi)
        FT.append(ft)
        GT.append(gt)
    return FT,GT

def claculate_phi_tao(X_t,Y_t,GT,FT):
    PHI_T = []
    TAO_T = []
    for time in range(t-1):
        phi_t = []
        tao_t = []
        for node in range(N):
            phi = X_t[time][node]*GT[time][node]
            tao = Y_t[time][node]-X_t[time][node]*FT[time][node]
            phi_t.append(phi)
            tao_t.append(tao)
        PHI_T.append(phi_t)
        TAO_T.append(tao_t)
    return PHI_T,TAO_T

def get_node(num):
    m = (num//N)-1
    n = num%N
    return m,n

def build_P_all_and_phi(phi_t, state_adj_set, node):
    P_all = np.empty((t-1, N), dtype=float)
    for tt in range(t-1):
        arr2 = np.array(state_adj_set[tt][2], dtype=float)
        arr3 = np.array(state_adj_set[tt][3], dtype=float)
        P_all[tt, :] = arr2 + arr3  

    phi_node = np.array([phi_t[tt][node] for tt in range(t-1)], dtype=float)

    return P_all, phi_node

def A1(node, PHI_T, state_adj_set):
    total_size = N*N + N

    P_all, phi_node = build_P_all_and_phi(PHI_T, state_adj_set, node)

    valid_single_idx = [i for i in range(N) if i != node]  
    valid_pair_nodes = []
    
    for orig in range(N, total_size):
        m, n = get_node(orig)
        if m != node and n != node and m != n:
            valid_pair_nodes.append((m, n))
    
    single_prods = P_all[:, valid_single_idx] if valid_single_idx else np.empty((len(phi_node), 0)) 
    
    if valid_pair_nodes:  
        m_indices, n_indices = zip(*valid_pair_nodes)
        pair_prods = P_all[:, m_indices] * P_all[:, n_indices]  
    else:
        pair_prods = np.empty((len(phi_node), 0))
    
    all_prods = np.hstack([single_prods, pair_prods])
    
    weighted_prods = all_prods * phi_node[:, np.newaxis]  
    result = all_prods.T @ weighted_prods 
    
    return result

def build_P_all_and_tao(tao_t, state_adj_set, node):
    P_all = np.empty((t-1, N), dtype=float)
    for tt in range(t-1):
        arr2 = np.array(state_adj_set[tt][2], dtype=float)
        arr3 = np.array(state_adj_set[tt][3], dtype=float)
        P_all[tt, :] = arr2 + arr3

    tao_node = np.array([tao_t[tt][node] for tt in range(t-1)], dtype=float)

    return P_all, tao_node

def A2(node, TAO_T, state_adj_set):
    total_size = N*N + N
    P_all, tao_node = build_P_all_and_tao(TAO_T, state_adj_set, node)

    valid_single_idx = [i for i in range(N) if i != node]
    valid_pair_nodes = []
    
    for orig in range(N, total_size):
        m2, n2 = get_node(orig)
        if m2 != node and n2 != node and m2 != n2:
            valid_pair_nodes.append((m2, n2))
    
    single_prods = P_all[:, valid_single_idx] if valid_single_idx else np.empty((len(tao_node), 0))
    
    if valid_pair_nodes:
        m_indices, n_indices = zip(*valid_pair_nodes)
        pair_prods = P_all[:, m_indices] * P_all[:, n_indices]
    else:
        pair_prods = np.empty((len(tao_node), 0))
    
    all_prods = np.hstack([single_prods, pair_prods])
    
    col_values = all_prods.T @ tao_node
    
    return col_values.reshape(-1, 1)

def claculate_RMN(node,PHI_T,TAO_T):
    A_left_U = A1(node, PHI_T, state_adj_set)
    A_right_U = A2(node, TAO_T, state_adj_set)
    x, residuals, rank, s = np.linalg.lstsq(A_left_U, A_right_U, rcond=None)
    return x


load_file = 'WS传播.pkl'
with open(load_file, 'rb') as f:
    loaded_data = pickle.load(f)
state_adj_set = loaded_data['state_adj'][200:]  

t=50000
X_t, Y_t = claculate_X_Y(state_adj_set)
THETA_t = claculate_theta(state_adj_set)
gamma = claculate_gamma(X_t, THETA_t, Y_t, initial_guess=0.9999)
FT, GT = claculate_F_G(gamma, THETA_t)
PHI_T, TAO_T = claculate_phi_tao(X_t, Y_t, GT, FT)

Aij_list = []
for node in range(N):
    aij = claculate_RMN(node, PHI_T, TAO_T)
    Aij_list.append(aij)
print(Aij_list)

import numpy as np
from itertools import combinations, permutations

def manual_confusion_matrix(y_pred,y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1)) 
    FP = np.sum((y_true == 0) & (y_pred == 1))  
    TN = np.sum((y_true == 0) & (y_pred == 0)) 
    FN = np.sum((y_true == 1) & (y_pred == 0)) 
    
    return TP, FP, TN, FN

def calculate_metrics(y_pred,y_true):
    TP, FP, TN, FN = manual_confusion_matrix(y_pred,y_true)
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics_dict = {
        'confusion_matrix': {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        },
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1

    }
    
    return metrics_dict

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

def evaluate_network(Aij_list,simplicial_matrix):
    scores = Aij_list
    labels = simplicial_matrix
    
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

def remove_diagonal(sub_adj_matrix):
    A = np.asarray(sub_adj_matrix)
    N = A.shape[0]
    assert A.shape[0] == A.shape[1]

    rows = []
    for i in range(N):
        new_row = np.delete(A[i], i, axis=0) 
        rows.append(new_row)

    B = np.vstack(rows)
    return B
def remove_two_simplicial(simplicial_matrix):
    index_list = []
    saved_elements = []
    total = N + N**2
    for node in range(N):
        valid_idx = []
        for orig in range(total):
            if orig < N:
                if orig != node:
                    valid_idx.append(orig)
            else:
                m, n = get_node(orig)
                if m != node and n != node and m != n:
                    valid_idx.append(orig)

        filtered = valid_idx[N+1:]
        index_list.append(filtered)
        row = simplicial_matrix[node]
        saved = [row[i] for i in filtered]
        saved_elements.append(saved)
    saved_elements = np.vstack(saved_elements)

    return saved_elements

def get_node_fix(num, N):
    adjusted_num = num - N
    m = adjusted_num // N
    n = adjusted_num % N
    return m, n
def valid_node(node, N):
    valid_pair_nodes = []
    for num in range(N, N**2 + N):
        m2, n2 = get_node_fix(num, N)
        if m2 != node and n2 != node and m2 != n2:
            valid_pair_nodes.append((m2, n2))
    return valid_pair_nodes
def create_complete_mapping_fixed(N):
    
    all_upper_triangular_pairs = []
    all_triangle_pairs = []
    node_pairs_info = {}
    
    for node in range(N):
        triangle_pairs = []
        valid_pairs = valid_node(node,N)
        upper_pairs = [(m, n) for m, n in valid_pairs if n < m < node]
        node_pairs_info[node] = {
            'valid_pairs': valid_pairs,
            'upper_pairs': upper_pairs
        }
        all_upper_triangular_pairs.extend([(node, m, n) for m, n in upper_pairs])
        triangle_pairs.extend([(node, m, n) for m, n in valid_pairs])
        all_triangle_pairs.append(triangle_pairs)

    

    start_col_index = N-1  
    position_to_col = {}  
    col_to_position = {}  
    for i in all_upper_triangular_pairs:
        position_key = i
        position_to_col[position_key] = all_triangle_pairs[position_key[0]].index(position_key)+start_col_index

    for node in range(N):
        for i in all_triangle_pairs[node]:
            position_key = i
            col_to_position[position_key] = all_triangle_pairs[position_key[0]].index(position_key)+start_col_index
        
    return all_upper_triangular_pairs,position_to_col,col_to_position,all_triangle_pairs

def get_triangle(Aij_list):
    one = Aij_list[:,:N-1]

    one_simplicial= np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    one_simplicial[mask] = one.flatten()

    n = one_simplicial.shape[0]
    result_dict = {}
    
    for triangle in combinations(range(n), 3):
        i, j, k = triangle
        
        edge_ij = one_simplicial[i, j]
        edge_jk = one_simplicial[j, k]
        edge_ik = one_simplicial[i, k]
        
        avg_value = (edge_ij + edge_jk + edge_ik) / 3
        
        for perm in permutations(triangle):
            result_dict[perm] = avg_value
    
    return result_dict,one_simplicial

def get_new_triangle(result_dict,Aij_list,col_to_position,a,b):
    final_list = Aij_list.copy()
    result_dict = {k: -v for k, v in result_dict.items()}
    for triangle, avg_value in result_dict.items():
        row = triangle[0]
        col = col_to_position[triangle]
        final_list[row][col] = a*avg_value + b*Aij_list[row][col] 
    return final_list

def count_continuous_nonzero(yy, zero_idx, tolerance=2):
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


def find_best_sep(y, sep_range, tolerance=2):
    ymin, ymax = np.min(y), np.max(y)
    best_sep = sep_range[0]
    best_score = (-1, -1)
    best_yy = None
    best_edges = None

    for sep in sep_range:
        yy, edges = np.histogram(y, bins=sep, range=(ymin, ymax))

        zero_idx = np.searchsorted(edges, 0.0, side='right') - 1
        zero_idx = np.clip(zero_idx, 0, sep - 1)

        count, total_freq = count_continuous_nonzero(yy, zero_idx, tolerance=tolerance)

        coverage = count / sep
        score = (coverage * math.sqrt(sep), total_freq)

        if score > best_score:
            best_score = score
            best_sep = sep
            best_yy = yy
            best_edges = edges

    return best_sep, best_yy, best_edges, best_score


def find_threshold(yy, edges, sep, simplicial=1):
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
    if simplicial == 1:
        valid = [idx for idx in sepid if bin_centers[idx] > 0]
        threshold = x[valid[0]]
    if simplicial == 2:
        valid = [idx for idx in sepid if bin_centers[idx] < 0]
        threshold = x[valid[-1]]
    return threshold


def estsr_single_cut(dtrs, sep_range=None, tolerance=2, simplicial=1):
    dtrs = np.array(dtrs).flatten()
    m = len(dtrs)
    xadj = np.zeros(m)

    y = dtrs
    if simplicial == 1:
        if sep_range is None:
            sep_range = range(3, 50)
        best_sep, best_yy, best_edges, best_score = find_best_sep(y, sep_range, tolerance=tolerance)
        threshold = find_threshold(best_yy, best_edges, best_sep, simplicial=simplicial)
        index = np.where(y >= threshold)[0]
        xadj[index] = 1

    if simplicial == 2:
        if sep_range is None:
            sep_range = range(3, 100)
        best_sep, best_yy, best_edges, best_score = find_best_sep(y, sep_range, tolerance=tolerance)
        threshold = find_threshold(best_yy, best_edges, best_sep, simplicial=simplicial)
        index = np.where(y <= threshold)[0]
        xadj[index] = 1

    return xadj


def process_array_list(array_list, sep_range=None, tolerance=2,simplicial=1):
    results = []
    for arr in array_list:
        xadj = estsr_single_cut(arr, sep_range=sep_range, tolerance=tolerance,simplicial=simplicial)
        results.append(xadj)
    return np.array(results)

beta1 = 0.05
beta2 = 0.5
beta = beta1
k = beta2/beta1     
log_factor_beta = np.log(1 - beta)        
log_factor_kbeta = np.log(1 - k * beta) 
Aij_list[:, :N-1] = Aij_list[:, :N-1] / log_factor_beta
Aij_list[:, N-1:] = Aij_list[:, N-1:] / log_factor_kbeta

all_upper_triangular_pairs, position_to_col,col_to_position,all_triangle_pairs = create_complete_mapping_fixed(N)
a = remove_diagonal(simplicial_matrix[:,:N])
b = remove_two_simplicial(simplicial_matrix)
new_matrix = np.hstack((a, b))

best_score_0 = -np.inf
best_a = 0
result_dict,one_simplicial = get_triangle(Aij_list)
for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    b = 1 - a
    final_list = get_new_triangle(result_dict, Aij_list, col_to_position, a, b)
    best_sep, best_yy, best_edges, best_score = find_best_sep(final_list[:, N-1:], range(3, 100), tolerance=0)
    if best_score[0] > best_score_0:
        best_score_0 = best_score[0]
        best_a = a

b=1-best_a
final_list = get_new_triangle(result_dict, Aij_list, col_to_position, best_a, b)
res_simplicial_one = process_array_list(Aij_list[:,:N-1], sep_range=range(3,50), tolerance=0, simplicial=1)
result_simplicial_one = evaluate_network(res_simplicial_one, new_matrix[:,:N-1])

res_simplicial_two = process_array_list(final_list[:,N-1:], sep_range=range(3,100), tolerance=0,simplicial=2)
result_simplicial_two = evaluate_network(res_simplicial_two, new_matrix[:,N-1:])

final_matrix = np.hstack((res_simplicial_one, res_simplicial_two))
result_all = evaluate_network(final_matrix, new_matrix)