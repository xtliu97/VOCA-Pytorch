# audio2face-pytorch

This repository provides PyTorch implementations for audio driven face meshes or blendshape models.  
Currently, it supports the following models:

- Audio2Face
- VOCA
- FaceFormer

And the following feature extractors are available:

- Wav2Vec
- MFCCExtractor

## Dataset

This repository uses VOCASET as the template, which is introduced in 'Capture, Learning, and Synthesis of 3D Speaking Styles' (CVPR 2019).  
Additionally, `FLAME_sample` has been extracted and converted to `assets/FLAME_sample.obj` and the Renderer has been redesigned. As a result, the `psbody` library is not required in this repository, which may cause installation issues for Apple Silicon users.

## License

VOCA [link](https://voca.is.tue.mpg.de/license.html)

## References

- VOCASET [ref](https://voca.is.tue.mpg.de)
- Cudeiro, Daniel, et al. "Capture, learning, and synthesis of 3D speaking styles." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. [ref](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cudeiro_Capture_Learning_and_Synthesis_of_3D_Speaking_Styles_CVPR_2019_paper.pdf)
- TimoBolkart/voca [ref](https://github.com/TimoBolkart/voca)
- Fan, Yingruo, et al. "FaceFormer: Speech-Driven 3D Facial Animation with Transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022. [ref](https://arxiv.org/abs/2112.05329)
- NVIDIA. Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion. [ref](https://research.nvidia.com/publication/2017-07_audio-driven-facial-animation-joint-end-end-learning-pose-and-emotion)
