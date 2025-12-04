# Third-Party Licenses

This project uses several third-party libraries and models. Below are the license details for each component.

## RT-DETR Model (via PaddleDetection)

**License:** Apache License 2.0

```
Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**Source:** https://github.com/PaddlePaddle/PaddleDetection

**Citation:**
```bibtex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection}, 
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

---

## Python Dependencies

### ONNXRuntime
**License:** MIT License  
**Source:** https://github.com/microsoft/onnxruntime

### OpenCV (opencv-python)
**License:** Apache License 2.0  
**Source:** https://github.com/opencv/opencv-python

### Supervision
**License:** MIT License  
**Source:** https://github.com/roboflow/supervision

### ByteTrack (via Supervision)
**License:** MIT License  
**Original Source:** https://github.com/ifzhang/ByteTrack

**Citation:**
```bibtex
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

### Pandas
**License:** BSD 3-Clause License  
**Source:** https://github.com/pandas-dev/pandas

### NumPy
**License:** BSD 3-Clause License  
**Source:** https://github.com/numpy/numpy

### NatSort
**License:** MIT License  
**Source:** https://github.com/SethMMorton/natsort

---

## Model Weights Attribution

The RT-DETR model weights used by this project are pre-trained on the COCO dataset:

**COCO Dataset:**
- License: Creative Commons Attribution 4.0 License
- Source: https://cocodataset.org/
- The model is trained on COCO, but we only use the model weights, not the dataset images

---

## License Compatibility

This project is licensed under the **MIT License**, which is compatible with:
- Apache 2.0 (RT-DETR, OpenCV) - Can be combined
- MIT (ONNXRuntime, Supervision, ByteTrack, NatSort) - Can be combined
- BSD 3-Clause (Pandas, NumPy) - Can be combined

**Combined License:** All components can be legally combined in this MIT-licensed project.

---

## Acknowledgments

We acknowledge the following projects and their contributors:

1. **PaddleDetection Team** - For RT-DETR implementation and weights
2. **ByteTrack Authors** - For the multi-object tracking algorithm
3. **Roboflow** - For the Supervision library
4. **Microsoft** - For ONNXRuntime inference engine
5. **OpenCV Community** - For computer vision utilities

---

## Updating This File

When adding new dependencies:
1. Check the license of the new dependency
2. Verify compatibility with MIT License
3. Add attribution here with source and license information
4. Update the requirements.txt with pinned versions
