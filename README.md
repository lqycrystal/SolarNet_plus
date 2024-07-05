# SolarNet_plus
This repository contains the code for our paper "Deep learning-based framework for city-scale rooftop solar potential estimation by considering roof superstructures".

## Introduction
Solar energy is an environmentally friendly energy source. Identifying suitable rooftops for solar panel installation contributes to not only sustainable energy plans but also carbon neutrality goals. Aerial imagery, bolstered by its growing availability, is a cost-effective data source for rooftop solar potential assessment at large scale. Existing studies generally do not take roof superstructures into account when determining how many solar panels can be installed. This procedure will lead to an overestimation of solar potential. Only several works have considered this issue, but none have devised a network that can simultaneously learn roof orientations and roof superstructures. Therefore, we devise SolarNet+, a novel framework to improve the precision of rooftop solar potential estimation. After implementing SolarNet+ on a benchmark dataset, we find that SolarNet+ outperforms other state-of-the-art approaches in both tasks - roof orientations and roof superstructure segmentation. Moreover, the SolarNet+ framework enables rooftop solar estimation at large-scale applications for investigating the correlation between urban rooftop solar potential and various local climate zone (LCZ) types. The results in the city of Brussels reveal that three specific LCZ urban types exhibit the highest rooftop solar potential efficiency: compact highrise (LCZ1), compact midrise (LCZ2), and heavy industry (LCZ10). The annual photovoltaic potential for these LCZ types is reported as 10.56 GWh/year/km^2, 11.77 GWh/year/km^2, and 10.70 GWh/year/km^2, respectively. 

## Datasets
We trained our networks on [RID dataset](https://mediatum.ub.tum.de/1655470).

Here, we regenerate the annotation mask that has three channels. The first channel is the roof segmentation map (2 classes), the second channel is the roof orientation map (6 classes), and the third channel is the roof superstructure map (Here, we use the original 9 classes. Actually, you can also combine all roof superstructures as one class, then you have 2 classes in total). The sample data are provided in the folder [`dlcode/wbf_data/train/seg/3_3_5.png`] 

## Deep Learning Network Training
Use [`dlcode/transolarnet.py`]  to train SolarNet+ on the RID dataset. 
> Note that some parameters can be adjusted according to your needs.
> This sample code is based on 9 classes of roof superstructures.
> You can get the trained model in the corresponding path after training.

## Deep Learning Network Evaluation
* To evaluate our SolarNet+ on small patch (e.g., with size of 512 x 512 pixels), use [`dlcode/predict_patch.py`] using the trained model.

* To evaluate our SolarNet+ on large image (e.g., with size of 5000 x 5000 pixels), use [`dlcode/predict_tif.py`] using the trained model.

> Note that some parameters can be adjusted according to your needs.
> This sample code is based on 9 classes of roof superstructures.
> The output files are the roof orientation map and the roof superstructure map.

## Rooftop Solar Potential Estimation

Use [`pvcode/cal_pv.py`]  to derive the rooftop solar potential using the roof orientation map and the roof superstructure map. 

> Note that some parameters can be adjusted according to your needs.
> This sample code is based on 2 classes of roof superstructures. If the roof superstructure map is 9 classes, you can first combine all roof superstructures as one class and then use this code.
> There are two output files. One is the CSV format of the PV generation of each roof segment; you can check the column of  electricity_generations  (unit is kWh/year). The other is the GEOJSON format of potential PV modules that can be installed in the future, and you can open it using QGIS.

## Acknowledgement
We appreciate the work from the following repository: [CGSANet](https://github.com/MrChen18/CGSANet)

## License
This code is available for non-commercial scientific research purposes under GNU General Public License v3.0. You agree to the terms in the LICENSE by downloading and using this code. Third-party datasets and software are subject to their respective licenses.

## Citations
> Qingyu Li, Sebastian Krapf, Lichao Mou, Yilei Shi, Xiao Xiang Zhu. Deep learning-based framework for city-scale rooftop solar potential estimation by considering roof superstructures. Applied Energy,2024.

