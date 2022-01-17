# DE-Opt-Main-Model

This implemention is for the paper work of "Toward Auto-learning Hyperparameters for deep learning-based Recommender Systems"
## Brief Introduction

This paper proposes a general hyperparameter optimization framework for existing DL-based RSs based on differential evolution (DE), named as DE-Opt. The main idea of DE-Opt is to incorporate DE into a DL-based recommendation model’s training process to auto-learn its hyperparameters λ (regulariza-tion coefficient) and η (learning rate) simultaneously at layer-granularity. Thereby, its performance of both recommendation accuracy and computa-tional efficiency is boosted. Empirical studies on three benchmark datasets verify that: 1) DE-Opt can significantly improve state-of-the-art DL-based recommendation models by making their λ and η adaptive, and 2) DE-Opt also significantly outperforms state-of-the-art DL-based recommendation models whose λ and/or η are/is adaptive. 
## Enviroment Requirement

The Code has been tested running under Python 3.7.0
requires:

tensorflow

numpy

scipy

sklearn

pandas

torch
## Dataset

This code contains Movilens 1M dataset for the example running`
filename: ratings.dat
format: user_ID::item_ID::ratings::time_stamp
Property: |U| 6040 |I| 3706	|YK| 1,000,209	Density 4.47%
Density denotes the percentage of known ratings or interactions in the dataset.
dataset is randomly split into three folds, i.e., training/validation/testing sets that contain 70%/10%/20% observed entries, respectively.
## Baseline Model
### Fixed Hyperparameter models && DE-Opt model
### rating prediction 
**AutoRec** && **AutoRec-(DE-Opt)**

Cite: Sedhain, S., Menon, A.K., Sanner, S., Xie, L.: Autorec: Autoencoders meet collabora-tive filtering. In: WWW, pp. 111–112 (2015)

**NRT** && **NRT-(DE-Opt)**

Cite: Li, P., Wang, Z., Ren, Z., Bing, L., Lam, W.: Neural rating regression with abstractive tips generation for recommendation. In: SIGIR, pp. 345–354 (2017)

**MetaMF** && **MetaMF-(DE-Opt)**

Cite: Lin, Y., Ren, P., Chen, Z., Ren, Z., Yu, D., Ma, J., Rijke, M.d., Cheng, X.: Meta ma-trix factorization for federated rating predictions. In: SIGIR,pp. 981–990 (2020)
### item ranking model
**NeuMF** && **NeuMF-(DE-Opt)**

Cite: He, X., Liao, L., Zhang, H., Nie, L., Hu, X., Chua, T.S.: Neural collaborative filtering. In: WWW. pp. 173–182 (2017)

**LRML** && **LRML-(DE-Opt)**  

Cite: Tay, Y., Anh Tuan, L., Hui, S.C.: Latent relational metric learning via memory-based attention for collaborative ranking. In: WWW, pp. 729–739 (2018)

**NGCF** && **NGCF-(DE-Opt)**

Cite: Wang, X., He, X., Wang, M., Feng, F., Chua, T.S.: Neural graph collaborative filtering. In: SIGIR, pp. 165–174 (2019)
### Adaptive Hyperparameter models && DE-Opt model
**SGDA**

Cite: Rendle, S.: Learning recommender systems with adaptive regularization. In: WSDM, pp. 133–142 (2012)

**SGD-λOpt** 

Cite: Chen, Y., Chen, B., He, X., Gao, C., Li, Y., Lou, J.G., Wang, Y.: λopt: Learn to regularize recommender models in finer levels. In: SIGKDD, pp. 978–986 (2019)

**Adam-λOpt**

Cite: Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR, (2015)

**AMSGrad-λOpt**

Cite: Reddi, S. J., Kale, S., & Kumar, S.: On the convergence of adam and beyond. In: ICLR, 2018

**AdaMod-λOpt**

Cite: Ding, J., Ren, X., Luo, R., Sun, X.: An adaptive and momental bound method for stochastic learning. arXiv preprint arXiv:1910.12249 (2019)

**DE-Opt**
