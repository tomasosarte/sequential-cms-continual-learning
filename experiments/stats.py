from scipy.stats import ttest_rel

def paired_ttest(baseline, cms):
    t_stat, p_value = ttest_rel(baseline, cms)
    return t_stat, p_value
