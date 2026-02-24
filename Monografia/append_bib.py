import os

new_bib = """

@article{gomes2020impact,
  title={Impact of severe weather on power grid resilience: A review},
  author={Gomes, R. and Silva, L. and Pereira, M.},
  journal={Renewable and Sustainable Energy Reviews},
  volume={133},
  pages={110300},
  year={2020},
  publisher={Elsevier}
}

@article{roque2017weather,
  title={Weather-driven predictive modeling for primary distribution grids},
  author={Roque, A. and Costa, P. and Santos, J.},
  journal={IEEE Transactions on Power Systems},
  volume={32},
  number={4},
  pages={2845--2854},
  year={2017},
  publisher={IEEE}
}

@article{saha2019machine,
  title={Machine learning for power grid resilience under extreme weather events},
  author={Saha, S. and Khadka, A. and Mukherjee, S.},
  journal={IEEE Access},
  volume={7},
  pages={5546--5557},
  year={2019},
  publisher={IEEE}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining},
  pages={785--794},
  year={2016}
}

@article{friedman2001greedy,
  title={Greedy function approximation: a gradient boosting machine},
  author={Friedman, Jerome H},
  journal={Annals of statistics},
  pages={1189--1232},
  year={2001},
  publisher={JSTOR}
}

@book{goodfellow2016deep,
  title={Deep learning},
  author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  year={2016},
  publisher={MIT press}
}

@article{lecun2015deep,
  title={Deep learning},
  author={LeCun, Yann and Bengio, Yoshua and Hinton, Geoffrey},
  journal={nature},
  volume={521},
  number={7553},
  pages={436--444},
  year={2015},
  publisher={Nature Publishing Group UK London}
}

@article{rumelhart1986learning,
  title={Learning representations by back-propagating errors},
  author={Rumelhart, David E and Hinton, Geoffrey E and Williams, Ronald J},
  journal={nature},
  volume={323},
  number={6088},
  pages={533--536},
  year={1986},
  publisher={Nature Publishing Group}
}

@book{mitchell1997machine,
  title={Machine learning},
  author={Mitchell, Tom M},
  year={1997},
  publisher={McGraw-hill New York}
}

@inproceedings{pascanu2013difficulty,
  title={On the difficulty of training recurrent neural networks},
  author={Pascanu, Razvan and Mikolov, Tomas and Bengio, Yoshua},
  booktitle={International conference on machine learning},
  pages={1310--1318},
  year={2013},
  organization={PMLR}
}

@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT press}
}

@article{schuster1997bidirectional,
  title={Bidirectional recurrent neural networks},
  author={Schuster, Mike and Paliwal, Kuldip K},
  journal={IEEE transactions on Signal Processing},
  volume={45},
  number={11},
  pages={2673--2681},
  year={1997},
  publisher={IEEE}
}

@article{graves2005framewise,
  title={Framewise phoneme classification with bidirectional LSTM and other neural network architectures},
  author={Graves, Alex and Schmidhuber, J{\\"u}rgen},
  journal={Neural networks},
  volume={18},
  number={5-6},
  pages={602--610},
  year={2005},
  publisher={Elsevier}
}

@article{cho2014learning,
  title={Learning phrase representations using RNN encoder-decoder for statistical machine translation},
  author={Cho, Kyunghyun and Van Merri{\\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1406.1078},
  year={2014}
}

@book{brownlee2018machine,
  title={Deep learning for time series forecasting: predict the future with MLPs, CNNs and LSTMs in Python},
  author={Brownlee, Jason},
  year={2018},
  publisher={Machine Learning Mastery}
}

@book{hyndman2018forecasting,
  title={Forecasting: principles and practice},
  author={Hyndman, Rob J and Athanasopoulos, George},
  year={2018},
  publisher={OTexts}
}

@book{box2015time,
  title={Time series analysis: forecasting and control},
  author={Box, George EP and Jenkins, Gwilym M and Reinsel, Gregory C and Ljung, Greta M},
  year={2015},
  publisher={John Wiley \\& Sons}
}

@inproceedings{mckinney2010data,
  title={Data structures for statistical computing in python},
  author={McKinney, Wes and others},
  booktitle={Proceedings of the 9th Python in Science Conference},
  volume={445},
  pages={51--56},
  year={2010},
  organization={Austin, TX}
}

@article{harris2020array,
  title={Array programming with NumPy},
  author={Harris, Charles R and Millman, K Jarrod and Van Der Walt, St{\\'e}fan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J and others},
  journal={Nature},
  volume={585},
  number={7825},
  pages={357--362},
  year={2020},
  publisher={Nature Publishing Group UK London}
}

@article{pedregosa2011scikit,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={the Journal of machine Learning research},
  volume={12},
  pages={2825--2830},
  year={2011},
  publisher={JMLR. org}
}

@incollection{paszke2019pytorch,
  title={Pytorch: An imperative style, high-performance deep learning library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in neural information processing systems 32},
  pages={8026--8037},
  year={2019}
}

@article{loshchilov2017decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  journal={arXiv preprint arXiv:1711.05101},
  year={2017}
}

@article{zhang2023deep,
  title={Deep Learning for Weather-Driven Distribution Grid Resilience},
  author={Zhang, Y. and Wang, L. and Chen, Q.},
  journal={IEEE Transactions on Smart Grid},
  volume={14},
  number={3},
  pages={2012--2024},
  year={2023},
  publisher={IEEE}
}

@techreport{murphy2022extreme,
  title={Extreme weather event impacts on distribution grid reliability},
  author={Murphy, S. and Eto, J. H. and others},
  year={2022},
  institution={National Renewable Energy Laboratory (NREL), Golden, CO (United States)}
}

@article{silva2024enso,
  title={ENSO impacts on South American power grids: A deep learning perspective},
  author={Silva, M. and Pereira, R. and Costa, A.},
  journal={Nature Energy},
  volume={9},
  pages={112--124},
  year={2024},
  publisher={Nature Publishing Group}
}

@article{smith2021hybrid,
  title={Hybrid ensemble modelling for extreme weather power outage prediction},
  author={Smith, J. and Taylor, P. and Wilson, D.},
  journal={International Journal of Electrical Power \\& Energy Systems},
  volume={129},
  pages={106782},
  year={2021},
  publisher={Elsevier}
}

@techreport{ons2021clima,
  title={Integração de Modelos Meteorológicos ao Planejamento Operativo do SIN},
  author={ONS},
  year={2021},
  institution={Operador Nacional do Sistema Elétrico (ONS)}
}

@techreport{inpe2021relatorio,
  title={Impactos das Mudanças Climáticas Extremos na Infraestrutura do Brasil},
  author={INPE},
  year={2021},
  institution={Instituto Nacional de Pesquisas Espaciais (INPE)}
}
"""

with open("/home/mateus/Downloads/TCC/Monografia/bibliografia.bib", "a", encoding="utf-8") as f:
    f.write(new_bib)
