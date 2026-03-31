## Conclusion and Direct Answers to Homework Questions

### 2) Weekly graph question (PCA scatter of PC1 vs PC2)

The three outliers in the upper-left corner have low PC1 scores (left side) and high PC2 scores (upper side). Based on the PCA loadings and how the synthetic data were generated, these points are most consistent with cases where series_3 is unusually low relative to series_1 and series_2.

It is hard to separate series_1 from series_2 in this PCA view because they were intentionally generated to be strongly related (series_2 is a noisy multiple of series_1 with small noise). Their shared variation is compressed mostly into PC1, while the larger independent noise in series_3 contributes more to the orthogonal direction (PC2) and creates visible outliers.

Advantages of the PC1/PC2 graph:
- Reduces 3 variables to 2 dimensions.
- Highlights unusual observations and dominant variation directions.

Disadvantages:
- Axes are linear combinations, not original variables, so interpretation is less direct.
- It can hide variable-specific details that are clearer in pairplots or correlation plots.

### 3) Working on the datasets (explicit conclusions)

Are the data what I expect and usable?
- The data are generally usable after cleaning and type conversion.
- Correlations and regressions show coherent structure rather than random noise.

Do I see outliers?
- Yes. Outliers appear in bubble plots and in the PCA score space, especially districts with large funding-gap magnitude and unusual demographic/funding combinations.

Does PCA suggest reduced dimensionality?
- Yes, partially. The first one to two principal components capture major shared variation in the numeric features and provide a compact summary for visualization.
- Some policy-relevant signals remain feature-specific, so PCA should complement (not replace) feature-level analysis.

Did I use correlation information to choose regression features?
- Yes. Regressions were chosen using high-correlation and policy-relevant pairs, including:
  - per_hispanic vs per_eng_learn
  - testscore_gap vs req_actual_diff
  - req_spending vs per_poverty
  - actual_spending vs req_spending

### 4) Storytelling With Data reproduction

I reproduced Figure 5.13 (page 148) with matching visual style choices (stacked bars, emphasized middle segments, custom annotation percentages, and muted/contrast color blocks). The goal was to match the original look and feel rather than exact source values, per the assignment instructions.
