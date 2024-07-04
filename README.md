# SolarNet_plus
## Datasets
We trained our networks on [RID dataset](https://mediatum.ub.tum.de/1655470).

Here, we regenerate the annotation mask that has three channels. The first channel is the roof segmentation map (2 classes), the second channel is the roof orientation map (6 classes), and the third channel is the roof superstructure map (9 classes). The sample data are provided in the folder [`dlcode/wbf_data/train/seg/3_3_5.png`] 

## Training
Use [`dlcode/transolarnet.py`]  to train SolarNet+ on the RID dataset. 
> Note that some parameters can be adjusted according to your needs.

## Evaluation
* To evaluate our SolarNet+ on small patch (e.g., with size of 512 x 512 pixels), use [`dlcode/predict_patch.py`]

* To evaluate our SolarNet+ on large image (e.g., with size of 5000 x 5000 pixels), use [`dlcode/predict_tif.py`]

## Acknowledgement
We appreciate the work from the following repository: [CGSANet](https://github.com/MrChen18/CGSANet)
## License
This code is available for non-commercial scientific research purposes under GNU General Public License v3.0. You agree to the terms in the LICENSE by downloading and using this code. Third-party datasets and software are subject to their respective licenses.
