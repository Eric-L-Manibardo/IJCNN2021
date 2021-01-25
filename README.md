# IJCNN-2021 Random Vector Functional Link Networks for Road Traffic Forecasting: Performance Comparison and Stability Analysis

This GitHub is intended to gather all datasets, Python source code, details on the hyper-parameter space considered for every model in the benchmark, and simulation results of the paper above.

# Abstract
Nowadays, Machine Learning algorithms benefit from a great momentum in multiple engineering and scientific fields. In the context of road traffic forecasting, the number of contributions resorting to these modeling techniques is increasing steadily over the last decades, in particular those based on Deep neural networks. In parallel, randomization based neural networks have progressively garnered much interest of the community due to their learning efficiency and good predictive performance level. Although these two properties are often sought for practical road traffic forecasting solutions, randomization based models have been investigated scarcely for this application. Indeed, the instability of these models due to the randomization of their learning phase is often a deciding factor for discarding them in favor of classical data based models. This research work sheds light on this matter by elaborating on the suitability of Random Vector Functional Link (RVFL) -- a representative randomization based neural network -- for road traffic forecasting. On one hand, multiple RVFL variants (single-layer RVFL, deep RVFL and ensemble deep RVFL) are compared to other Machine Learning algorithms over an extensive experimental setup,  which comprises the prediction of traffic data collected at diverse geographical locations that differ in the context and nature of traffic measurements. On the other hand, the stability of RVFL models is subsequently analyzed towards providing insights about the compromise between model complexity and performance. The results obtained by the distinct RVFL based architectures, are found to be similar than those elicited by other data driven methods, yet requiring a much lower number of parameters and thereby, drastically shorter training times and computational effort.

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
https://arxiv.org/abs/2012.02260
###### BibTex:
```
NOT AVAILABLE YET
```
#### Note: These datasets should be used for research only.



