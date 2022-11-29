# Image-Matching-Paper-List
A personal list of papers and resources for image matching and pose estimation, including perspective images and panoramas (marked with :globe_with_meridians:).

---
- [Detector-free matching](#detector-free-matching)
- [Detector-based matching](#detector-based-matching)
   - [Keypoints and descriptors](#keypoints-and-descriptors)
   - [Feature matching](#feature-matching)
- [Pose estimation](#pose-estimation)
- [Datasets](#datasets)
- [Challenges and workshops](#challenges-and-workshops)
---

## Detector-free matching
* Patch2Pix: Epipolar-Guided Pixel-Level Correspondences [[CVPR 2021](https://arxiv.org/pdf/2012.01909.pdf)] [[patch2pix](https://github.com/GrumpyZhou/patch2pix)]
* LoFTR: Detector-Free Local Feature Matching with Transformers [[CVPR 2021](https://arxiv.org/pdf/2104.00680.pdf)] [[LoFTR](https://github.com/zju3dv/LoFTR)]
* A case for using rotation invariant features in state of the art feature matchers [[CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/IMW/papers/Bokman_A_Case_for_Using_Rotation_Invariant_Features_in_State_of_CVPRW_2022_paper.pdf)] [[se2-loftr](https://github.com/inkyusa/se2-loftr)]
* Local Feature Matching with Transformers for low-end devices [[arXiv 2022](https://arxiv.org/pdf/2202.00770.pdf)] [[Coarse_LoFTR_TRT](https://github.com/Kolkir/Coarse_LoFTR_TRT)]
* COTR: Correspondence Transformer for Matching Across Images [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_COTR_Correspondence_Transformer_for_Matching_Across_Images_ICCV_2021_paper.pdf)] [[COTR](https://github.com/ubc-vision/COTR)]
* ECO-TR: Efficient Correspondences Finding Via Coarse-to-Fine Refinement [[ECCV 2022](https://arxiv.org/pdf/2209.12213.pdf)] [[ECO-TR](https://github.com/dltan7/ECO-TR)]
* QuadTree Attention for Vision Transformers [[ICLR 2022](https://arxiv.org/pdf/2201.02767.pdf)] [[QuadTreeAttention](https://github.com/Tangshitao/QuadTreeAttention)]
* Deep Kernelized Dense Geometric Matching [[arXiv 2022](https://arxiv.org/pdf/2202.00667.pdf)] [[DKM](https://github.com/Parskatt/DKM)]
* ASpaFormer: Detector-Free Matching with Adaptive Span Transformer [[ECCV 2022](https://arxiv.org/pdf/2208.14201.pdf)] [[aspanformer-initial-release](https://github.com/slyxsw/aspanformer-initial-release)]
* TopicFM: Robust and Interpretable Topic-Assisted Feature Matching [[AAAI 2023](https://arxiv.org/pdf/2207.00328.pdf)] [[TopicFM](https://github.com/TruongKhang/TopicFM)]


---
## Detector-based matching
### Keypoints and descriptors
* :globe_with_meridians: SPHORB: A Fast and Robust Binary Feature on the Sphere [[IJCV 2015](http://cic.tju.edu.cn/faculty/lwan/paper/SPHORB/SPHORB.html)] [[SPHORB](https://github.com/tdsuper/SPHORB)]
* SuperPoint: Self-Supervised Interest Point Detection and Description [[CVPRW 2018](https://arxiv.org/pdf/1712.07629.pdf)] [[SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork)]
* ASLFeat: Learning Local Features of Accurate Shape and Localization [[CVPR 2020](https://arxiv.org/pdf/2003.10071.pdf)] [[ASLFeat](https://github.com/lzx551402/ASLFeat)]
* DISK: Learning local features with policy gradient [[NeurIPS 2020](https://arxiv.org/pdf/2006.13566.pdf)] [[disk](https://github.com/cvlab-epfl/disk)]
* ALIKE: Accurate and Lightweight Keypoint Detection and Descriptor Extraction [[TMM 2022](https://arxiv.org/pdf/2112.02906.pdf)] [[ALIKE](https://github.com/Shiaoming/ALIKE)]

### Feature matching
* Learning Two-View Correspondences and Geometry Using Order-Aware Network [[ICCV 2019](https://arxiv.org/pdf/1908.04964.pdf)] [[OANet](https://github.com/zjhthu/OANet)]
* SuperGlue: Learning Feature Matching with Graph Neural Networks [[CVPR 2020](https://arxiv.org/pdf/1911.11763.pdf)] [[SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)]
* Learning to Match Features with Seeded Graph Matching Network [[ICCV 2021](https://arxiv.org/pdf/2108.08771.pdf)] [[SGMNet](https://github.com/vdvchen/SGMNet)]

---
## Pose estimation
* Wide-Baseline Relative Camera Pose Estimation with Directional Learning [[CVPR 2021](https://arxiv.org/pdf/2106.03336.pdf)] [[DirectionNet](https://github.com/arthurchen0518/DirectionNet)]
* :globe_with_meridians: Robust 360-8PA: Redesigning The Normalized 8-point Algorithm for 360-FoV Images [[ICRA 2021](https://arxiv.org/pdf/2104.10900.pdf)] [[robust_360_8PA](https://github.com/EnriqueSolarte/robust_360_8PA)]
* :globe_with_meridians: Pose Estimation for Two-View Panoramas: a Comparative Analysis [[CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/OmniCV/papers/Murrugarra-Llerena_Pose_Estimation_for_Two-View_Panoramas_Based_on_Keypoint_Matching_A_CVPRW_2022_paper.pdf)] [[Keypoints](https://github.com/Artcs1/Keypoints)]
* The 8-Point Algorithm as an Inductive Bias for Relative Pose Prediction by ViTs [[3DV 2022](https://crockwell.github.io/rel_pose/data/paper.pdf)] [[rel_pose](https://github.com/crockwell/rel_pose)]
* End2End Multi-View Feature Matching using Differentiable Pose Optimization [[arXiv 2022](https://arxiv.org/pdf/2205.01694.pdf)] [[]()]
* :globe_with_meridians: CoVisPose: Co-visibility Pose Transformer for Wide-Baseline Relative Pose Estimation in 360  Indoor Panoramas [[ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920610.pdf)] [[]()]
* :globe_with_meridians: GPR-Net: Multi-view Layout Estimation via a Geometry-aware Panorama Registration Network [[arXiv 2022](https://arxiv.org/pdf/2210.11419.pdf)] [[]()]

---
## Datasets
* [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
* [ScanNet](http://www.scan-net.org/)
* :globe_with_meridians: [Matterport3D](https://niessner.github.io/Matterport/)
* :globe_with_meridians: [Zillow Indoor Dataset (ZInD)](https://github.com/zillow/zind)

---
## Challenges and workshops
* [Image Matching Challenge 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview)
* [Image Matching Challenge 2021](https://www.cs.ubc.ca/research/image-matching-challenge/current/)
* [Image Matching Challenge 2020](https://www.cs.ubc.ca/research/image-matching-challenge/2020/)
* [Image Matching Challenge 2019](https://image-matching-workshop.github.io/leaderboard/)
* [Image Matching: Local Features and Beyond workshop at CVPR](https://image-matching-workshop.github.io/)
* :globe_with_meridians: [Omnidirectional Computer Vision workshop at CVPR](https://sites.google.com/view/omnicv2022)

---
Format:
* Title [[journal & year]()] [[repo]()]
