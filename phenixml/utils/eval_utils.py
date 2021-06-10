import numpy as np

def tukey_test(values):
  q1 = np.quantile(values,0.25)
  q3 = np.quantile(values,0.75)
  iqr = q3-q1
  inner_fence = 1.5*iqr
  outer_fence = 3*iqr

  #inner fence lower and upper end
  inner_fence_le = q1-inner_fence
  inner_fence_ue = q3+inner_fence

  #outer fence lower and upper end
  outer_fence_le = q1-outer_fence
  outer_fence_ue = q3+outer_fence

  outliers_prob = np.sort(np.concatenate([np.where(values<=outer_fence_le)[0],np.where(values>=outer_fence_ue)[0]]))
  outliers_poss = np.sort(np.concatenate([np.where(values<=inner_fence_le)[0],np.where(values>=inner_fence_ue)[0]]))

  return outliers_prob,outliers_poss
