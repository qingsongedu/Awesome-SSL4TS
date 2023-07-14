# Awesome Self-Supervised Learning for Time Series (SSL4TS)


[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/qingsongedu/awesome-self-supervised-learning-timeseries)
[![Visits Badge](https://badges.pufler.dev/visits/qingsongedu/awesome-self-supervised-learning-timeseries)](https://badges.pufler.dev/visits/qingsongedu/awesome-self-supervised-learning-timeseries)
<!-- ![Forks](https://img.shields.io/github/forks/qingsongedu/awesome-self-supervised-learning-timeseries) -->


A professionally curated list of awesome resources (paper, code, data, etc.) on **Self-Supervised Learning for Time Series (SSL4TS)**, which is the first work to comprehensively and systematically summarize the recent advances of Self-Supervised Learning for modeling time series data to the best of our knowledge.

We will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

For general **AI for Time Series (AI4TS)** Papers, Tutorials, and Surveys at the **Top AI Conferences and Journals**, please check [This Repo](https://github.com/qingsongedu/awesome-AI-for-time-series-papers). 

 
## Survey paper

[**Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects**](https://arxiv.org/abs/2306.10125)  

Kexin Zhang, [Qingsong Wen](https://sites.google.com/site/qingsongwen8/), Chaoli Zhang, Rongyao Cai, Ming Jin, Yong Liu, James Zhang, [Yuxuan Liang](https://yuxuanliang.com/), [Guansong Pang](https://sites.google.com/site/gspangsite), [Dongjin Song](https://songdj.github.io/), [Shirui Pan](https://shiruipan.github.io/).

#### If you find this repository helpful for your work, please kindly cite our survey paper.

```bibtex
@article{zhang2023ssl4ts,
  title={Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects},
  author={Kexin Zhang and Qingsong Wen and Chaoli Zhang and Rongyao Cai and Ming Jin and Yong Liu and James Zhang and Yuxuan Liang and Guansong Pang and Dongjin Song and Shirui Pan}
  journal={arXiv preprint arXiv:2306.10125},
  year={2023}
}
```

## Taxonomy of Self-Supervised Learning for Time Series
<!-- ![xxx](SSL4TS_taxonomy.jpg) -->
<img src="SSL4TS_taxonomy.jpg" width=900 align=middle> <br />

<!-- ![xxx](generative_adversarial_ssl4ts.jpg) -->
<img src="generative_adversarial_ssl4ts.jpg" width=900 align=middle> <br />

<!-- ![xxx](contrastive_ssl4ts.jpg) -->
<img src="contrastive_ssl4ts.jpg" width=900 align=middle> <br />



## Category of Self-Supervised Learning for Time Series

### Generative-based Methods on SSL4TS
#### Autoregressive-based forecasting

- Timeseries anomaly detection using temporal hierarchical one-class network, NeurIPS2020.
- Self-supervised transformer for sparse and irregularly sampled multivariate clinical time-series,
- Graph neural network-based anomaly detection in multivariate time series, AAAI
- Semisupervised time series classification model with self-supervised learning,

#### Autoencoder-based reconstruction

- xxx

#### Diffusion-based generation

### Contrastive-based Methods on SSL4TS
#### Sampling contrast
#### Prediction contrast
#### Augmentation contrast
#### Prototype contrast
#### Expert knowledge contrast

### Adversarial-based Methods on SSL4TS

#### Time series generation and imputation
#### Auxiliary representation enhancement


## Applications and Datasets on SSL4TS
#### Anomaly Detection
|Dataset|Size|Dimension|Source|Link|Comment|
|:---:|:---:|:---:|:---:|:---:|:---|
|PSM|132,481 / 87,841|26|[[paper]](https://dl.acm.org/doi/10.1145/3447548.3467174)|[[link]](https://github.com/eBay/RANSynCoders)|AnRa: 27.80%|
|SMD|708,405 / 708,405|38|[[paper]](https://dl.acm.org/doi/10.1145/3292500.3330672)|[[link]](https://github.com/NetManAIOps/OmniAnomaly)|AnRa: 4.16%|
|MSL|58,317 / 73,729|55|[[paper]](https://dl.acm.org/doi/10.1145/3219819.3219845)|[[link]](https://github.com/khundman/telemanom)|AnRa: 10.72%|
|SMAP|135,183 / 427,617|25|[[paper]](https://dl.acm.org/doi/10.1145/3219819.3219845)|[[link]](https://github.com/khundman/telemanom)|AnRa: 13.13%|
|SWaT|475,200 / 449,919|51|[[paper]](https://link.springer.com/chapter/10.1007/978-3-319-71368-7_8)|[[link]](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)|AnRa: 12.98%|
|WADI|1,048,571 / 172,801|103|[[paper]](https://dl.acm.org/doi/10.1145/3055366.3055375)|[[link]](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)|AnRa: 5.99%|

#### Forecasting
|Dataset|Size|Dimension|Source|Link|Comment|
|:---:|:---:|:---:|:---:|:---:|:---|
|ETTh|17,420|7|[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17325)|[[link]](https://github.com/zhouhaoyi/ETDataset)|SaIn: 1h|
|ETTm|69,680|7|[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17325)|[[link]](https://github.com/zhouhaoyi/ETDataset)|SaIn: 15min|
|Wind|10,957|28|*Non*|[[link]](https://www.kaggle.com/datasets/sohier/30-years-of-european-wind-generation)|SaIn: 1day|
|Electricity|26,304|321|*Non*|[[link]](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)|SaIn: 1hour|
|ILI|966|7|*Non*|[[link]](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)|SaIn: 1weak|
|Weather|52,696|21|*Non*|[[link]](https://www.bgc-jena.mpg.de/wetter/)|SaIn: 10min|
|Traffic|17,544|862|*Non*|[[link]](http://pems.dot.ca.gov/)|SaIn: 1hour|
|Exchange|7,588|8|[[paper]](https://dl.acm.org/doi/abs/10.1145/3209978.3210006)|[[link]](https://github.com/laiguokun/LSTNet)|SaIn: 1day|
|Solar|52,560|137|*Non*|[[link]](https://www.nrel.gov/grid/solar-power-data.html)|SaIn: 10min|

#### Classification and Clustering
|Dataset|Size|Dimension|Source|Link|Comment|
|:---:|:---:|:---:|:---:|:---:|:---|
|HAR|17,3056 / 173,056|9|[[paper]](https://web.archive.org/web/20210506171341id_/https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-84.pdf)|[[link]](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)|Classes: 6|
|UCR 130|128*M|1|[[paper]](https://arxiv.org/pdf/1810.07758.pdf)|[[link]](https://www.cs.ucr.edu/~eamonn/time_series_data/)|N/A|
|UEA 30|30*M|D|[[paper]](https://arxiv.org/abs/1811.00075)|[[link]](https://www.timeseriesclassification.com/)|N/A|



## Time Series Related Survey
* Transformers in Time Series: A Survey, in *IJCAI* 2023. [[paper]](https://arxiv.org/abs/2202.07125) [[link]](https://github.com/qingsongedu/time-series-transformers-review)
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2002.12478)
* Neural temporal point processes: a review, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2104.03528v5)
* Time-series forecasting with deep learning: a survey, in *Philosophical Transactions of the Royal Society A* 2021. [\[paper\]](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0209)
* Deep learning for time series forecasting: a survey, in *Big Data* 2021. [\[paper\]](https://www.liebertpub.com/doi/abs/10.1089/big.2020.0159)
* Neural forecasting: Introduction and literature overview, in *arXiv* 2020. [\[paper\]](https://arxiv.org/abs/2004.10240) 
* Deep learning for anomaly detection in time-series data: review, analysis, and guidelines, in *Access* 2021. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/9523565) 
* A review on outlier/anomaly detection in time series data, in *ACM Computing Surveys* 2021. [\[paper\]](https://arxiv.org/abs/2002.04236)
* A unifying review of deep and shallow anomaly detection, in *Proceedings of the IEEE* 2021. [\[paper\]](http://128.84.4.34/abs/2009.11732)
* Deep learning for time series classification: a review, in *Data Mining and Knowledge Discovery* 2019. [\[paper\]](https://link.springer.com/article/10.1007/s10618-019-00619-1?sap-outbound-id=11FC28E054C1A9EB6F54F987D4B526A6EE3495FD&mkt-key=005056A5C6311EE999A3A1E864CDA986)
* More related time series surveys, tutorials, and papers can be found at this [repo](https://github.com/qingsongedu/awesome-AI-for-time-series-papers).

## Self-Supervised Learning Tutorial/Survey in Other Disciplines
* A cookbook of self-supervised learning, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.12210)
* Self-supervised Learning: Generative or Contrastive, in *TKDE* 2021. [\[paper\]](https://arxiv.org/abs/2006.08218)
















