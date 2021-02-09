from scipy import stats
import numpy as np


def contingency_coefficient(contingency_table):
    """still on development
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, c = contingency_table.shape
    k = min(r, c)
    devider = ((r - 1) / r * (c - 1) / c) ** (1 / 4)
    cc = (chi2 / (n + chi2)) ** (1 / 2)
    v = (chi2 / (n * (k - 1))) ** (1 / 2)
    return cc, v, cc / devider


def cramers_v(contingency_table):
    """cramers v with bias correction
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def odds_ratio(contingency_table):
    oddsratio, pvalue = stats.fisher_exact(contingency_table)
    return oddsratio
