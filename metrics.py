import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import torch
from sklearn.linear_model import LogisticRegression
from math import pi, sqrt



# Metric is Demographic Parity (DP1). Code source https://github.com/steven7woo/fair_regression_reduction/tree/master based on: Fair Regression: Quantitative Definitions and Reduction-based Algorithms https://arxiv.org/abs/1905.12843

def get_histogram(pred, theta_indices):
    hist, _ = np.histogram(pred, bins=np.append(theta_indices, theta_indices.iloc[-1] + 1))
    return pd.Series(hist, index=theta_indices)

def calc_demographic_parity_disparity(pred, sensitive_features):
    Theta = np.unique(pred)  
    theta_indices = pd.Series(Theta)

    histogram_all = get_histogram(pred, theta_indices)
    total_count = histogram_all.sum()
    PMF_all = histogram_all / total_count

    max_DP_disp = 0
    for g in np.unique(sensitive_features):
        histogram_g = get_histogram(pred[sensitive_features == g], theta_indices)
        PMF_g = histogram_g / histogram_g.sum()
        max_DP_disp = max(max_DP_disp, np.max(np.abs(np.cumsum(PMF_all) - np.cumsum(PMF_g))))

    return max_DP_disp




# Metrics are independence and separation. Code source https://dalex.drwhy.ai/python-dalex-fairness-regression.html based on: Fairness Measures for Regression via Probabilistic Classification https://arxiv.org/pdf/2001.06089.pdf

def calculate_regression_measures(y, y_hat, protected, privileged):    
    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])
    for unprivileged in unique_unprivileged:
        # filter elements
        array_elements = np.isin(protected, [privileged, unprivileged])

        y_u = ((y[array_elements] - y[array_elements].mean()) / y[array_elements].std()).reshape(-1, 1)
        s_u = ((y_hat[array_elements] - y_hat[array_elements].mean()) / y_hat[array_elements].std()).reshape(-1, 1)

        #y_u = np.array(y[array_elements]).reshape(-1, 1)
        #s_u = np.array(y_hat[array_elements]).reshape(-1, 1)


        a = np.where(protected[array_elements] == privileged, 1, 0)

        p_s = LogisticRegression()
        p_ys = LogisticRegression()
        p_y = LogisticRegression()
    
        p_s.fit(s_u, a)
        p_y.fit(y_u, a)
        p_ys.fit(np.c_[y_u, s_u], a)
        pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
        pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
        pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]
        
        n = len(a)
    
        r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
        r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
        r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()
    
        to_append = pd.DataFrame({'subgroup': [unprivileged],
                                'independence': [r_ind],
                                'separation': [r_sep],
                                'sufficiency': [r_suf]})

        data = pd.concat([data, to_append])

    to_append = pd.DataFrame({'subgroup': [privileged],
                            'independence': [1],
                            'separation': [1],
                            'sufficiency': [1]})

    data.index = data.subgroup
    data = data.iloc[:, 1:]
    return data




# Metric is Demographic Parity with Wasserstein Barycenters  (DP2). Code source https://github.com/lucaoneto/NIPS2020_Fairness based on: Fair Regression with Wasserstein Barycenters https://arxiv.org/pdf/2006.07286


def optimized_f_fai(Y, S):
    vv = np.unique(S)
    nn = [np.sum(S == v) for v in vv]
    Y_subsets = [Y[S == v] for v in vv]
    tt = np.linspace(min(Y), max(Y), 1000)
    sorted_subsets = [np.sort(subset) for subset in Y_subsets]
    
    cumulative_counts = [np.searchsorted(subset, tt, side='right') for subset in sorted_subsets]
    
    cdf_values = [counts / n for counts, n in zip(cumulative_counts, nn)]
    differences = np.abs(cdf_values[0] - cdf_values[1])
    fai = np.max(differences)
    
    return fai



# Metric is Demographic Parity with Renyi correlation (DP3). Code source https://github.com/criteo-research/continuous-fairness based on: Fairness-Aware Learning for Continuous Attributes and Treatments https://proceedings.mlr.press/v97/mary19a/mary19a.pdf

class kde:
    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.train_x = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]
        assert d == self.d

        data = x.unsqueeze(-2)

        train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        pdf_values = (
                         torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / self.bandwidth

        return pdf_values

def _unsqueeze_multiple_times(input, axis, times):

    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output

def _joint_2(X, Y, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
    joint_density = density(data)

    nbins = int(min(50, 5. / joint_density.std))
    #nbins = np.sqrt( Y.size/5 )
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)

    xx, yy = torch.meshgrid([x_centers, y_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
    h2d = joint_density.pdf(grid) + damping
    h2d /= h2d.sum()
    return h2d


def hgr(X, Y, damping = 1e-10):

    h2d = _joint_2(X, Y, kde, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return torch.svd(Q)[1][1]


def _joint_3(X, Y, Z, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    Z = (Z - Z.mean()) / Z.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], -1)
    joint_density = density(data)  # + damping

    nbins = int(min(50, 5. / joint_density.std))
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)
    z_centers = torch.linspace(-2.5, 2.5, nbins)
    xx, yy, zz = torch.meshgrid([x_centers, y_centers, z_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], -1)

    h3d = joint_density.pdf(grid) + damping
    h3d /= h3d.sum()
    return h3d


def hgr_cond(X, Y, Z):
    damping = 1e-10
    h3d = _joint_3(X, Y, Z, kde, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return np.array(([torch.svd(Q[:, :, i])[1][1] for i in range(Q.shape[2])]))

