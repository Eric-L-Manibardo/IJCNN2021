# IJCNN-2021 Random Vector Functional Link Networks for Road Traffic Forecasting: Performance Comparison and Stability Analysis

This GitHub is intended to gather all datasets, Python source code, details on the hyper-parameter space considered for every model in the benchmark, and simulation results of the paper above.

# Proyect structure
Bearing in mind that the main goals of this study gravitate around randomization based neural networks, we have split the experimental set up in two folders: one for randomization and the other for the rest of the considered learning methods. Once inside the folder, both shares the same structure of 3 steps:

1. Randomization_expsetup
   1. Madrid_code
   2. NYC_code
   3. PeMS_code
   4. Seattle_code
      1. Validation
         - Searching the best hyperparameter configuration for each model and scenario.
      2. Test
         - Computing the test score for each model and scenario, using the architecture configuration set in previous stage.
      3. Stability (only for randomization neural networks of multiple layers)
         - Analyzing the dispersion of the test results for specific architectures over 100 test runs.
2. Other_methods_expsetup
3. Plot
   - Code to generate the figures of the paper.


# Abstract
Nowadays, Machine Learning algorithms enjoy a great momentum in multiple engineering and scientific fields. In the context of road traffic forecasting, the number of contributions resorting to these modeling techniques is increasing steadily over the last decade, in particular those based on deep neural networks. In parallel, randomization based neural networks have progressively garnered the interest of the community due to their learning efficiency and competitive predictive performance. Although these two properties are often sought for practical traffic forecasting solutions, randomization based neural networks have so far been scarcely investigated for this domain. In particular, the instability of these models due to the randomization of part of their parameters is often a deciding factor for discarding them in favor of other modeling choices. This research work sheds light on this matter by elaborating on the suitability of Random Vector Functional Link (RVFL) for road traffic forecasting. On one hand, multiple RVFL variants (single-layer RVFL, deep RVFL and ensemble deep RVFL) are compared to other Machine Learning algorithms over an extensive experimental setup, which comprises traffic data collected at diverse geographical locations that differ in the context and nature of the collected traffic measurements. On the other hand, the stability of RVFL models is analyzed towards providing insights about the compromise between model complexity and performance. The results obtained by the distinct RVFL approaches are found to be similar than those elicited by other data driven methods, yet requiring a much lower number of trainable parameters and thereby, drastically shorter training times and computational effort.

# Dataset overview

| Location         | Data nature |  Scope  | Sensor type | Time resolution | Year |                              Data source                              |
|------------------|:-----------:|:-------:|:-----------:|:---------------:|:----:|:---------------------------------------------------------------------:|
| Madrid city      |     Flow    |  Urban  |   Roadside  |        15       | 2018 | https://datos.madrid.es/portal/site/egob/                             |
| California state |     Flow    | Freeway |   Roadside  |        5        | 2017 | http://pems.dot.ca.gov/                                               |
| New York city    |    Speed    |  Urban  |   Roadside  |        5        | 2016 | https://www.kaggle.com/crailtap/nyc-real-time-traffic-speed-data-feed |
| Seattle city     |    Speed    | Freeway |   Roadside  |        5        | 2015 | https://github.com/zhiyongc/Seattle-Loop-Data                         |

---
#### If you use any dataset in your work, please cite the following reference:
###### Reference:
NOT AVAILABLE YET
###### BibTex:
```
NOT AVAILABLE YET
```
#### Note: These datasets should be used for research only.



