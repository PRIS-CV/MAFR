# MAFR
Code release for "Multi-View Active Fine-Grained Visual Recognition" (ICCV 2023）

<img src="setting_figure.pdf" width="500"/>

**Abstract**: Despite the remarkable progress of Fine-grained visual classification (FGVC) with years of history, it is still limited to recognizing 2D images. Recognizing objects in the physical world (i.e., 3D environment) poses a unique challenge -- discriminative information is not only present in visible local regions but also in other unseen views. Therefore, in addition to finding the distinguishable part from the current view, efficient and accurate recognition requires inferring the critical perspective with minimal glances. E.g., a person might recognize a "Ford sedan" with a glance at its side and then know that looking at the front can help tell which model it is. In this paper, towards FGVC in the real physical world, we put forward the problem of multi-view active fine-grained visual recognition (MAFR) and complete this study in three steps: (i) a multi-view, fine-grained vehicle dataset is collected as the testbed, (ii) a pilot experiment is designed to validate the need and research value of MAFR, (iii) a policy-gradient-based framework along with a dynamic exiting strategy is proposed to achieve efficient recognition with active view selection. Our comprehensive experiments demonstrate that the proposed method outperforms previous multi-view recognition works and can extend existing state-of-the-art FGVC methods and advanced neural networks to become FGVC experts in the 3D environment.


## Requirements
- python 3.8
- CUDA 10.2
- PyTorch 1.10.0
- torchvision 0.11.1


## Training
You may launch the program with `train.sh`


## Citation
If you find our idea or some of our codes useful in your research, please consider citing:
```
@InProceedings{du2023multi,
  title={Multi-View Active Fine-Grained Visual Recognition},
  author={Du, Ruoyi and Yu, Wenqing and Wang, Heqing and Lin, Ting-En and Chang, Dongliang and Ma, Zhanyu},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```


## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- duruoyi@bupt.edu.cn
- changdongliang@pris-cv.cn
- mazhanyu@bupt.edu.cn
