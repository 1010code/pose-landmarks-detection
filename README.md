
## MediaPipe Pose

|    Origin    | Pose landmark |    3D landmarks        |
|--------|---------------|---------------|
|   ![](./image/test1.png)  |  ![](./screenshot/output.jpg)  |[![](./screenshot/output3d.png)](https://1010code.github.io/pose-landmarks-detection/screenshot/output.html)|



https://github.com/1010code/pose-landmarks-detection/assets/20473922/ae0511ec-6cfa-4d29-896b-c561d9695360


## OpenPose
- Body25
  - pose_deploy.prototxt: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/body_25/pose_deploy.prototxt 
  - pose_iter_584000: https://www.dropbox.com/s/3x0xambj2rkyrap/pose_iter_584000.caffemodel

- MPI
  - pose_deploy_linevec_faster_4_stages.prototxt: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt
  - pose_iter_116000.caffemodel: https://www.dropbox.com/s/d08srojpvwnk252/pose_iter_116000.caffemodel

- COCO
  - https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt
  - [pose_iter_102000.caffemodel: https://www.dropbox.com/s/gqgsme6sgoo0zxf/pose_iter_102000.caffemodel](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/blob/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel)

## Reference
- [OpenPose 基於OpenCV DNN 的單人姿態估計](https://www.aiuai.cn/aifarm943.html)
- [AlphaPose](https://github.com/Fang-Haoshu/Halpe-FullBody)
- [OpenPose vs. AlphaPose](https://blog.songhaban.com/2022/02/openpose-vs-alphapose-which-one-is.html)
- [mmpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
- [角度量測1](https://github.com/mansikataria/SquatDetection/tree/main)
- [角度量測2](https://learnopencv.com/ai-fitness-trainer-using-mediapipe/)