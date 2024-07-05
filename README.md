# SolarNet_plus
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

