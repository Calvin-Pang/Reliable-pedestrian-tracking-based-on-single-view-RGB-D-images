# Reliable pedestrian tracking based on single view RGB-D images
This is the code of my graduation project **Reliable pedestrian tracking based on single view RGB-D images**. My work is mainly based on the paper of Daniil Osokin **Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose**. Here is the Github of this great work [**Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose**](https://github.com/Calvin-Pang/lightweight-human-pose-estimation.pytorch).

Mainly, in my project I explored the design of several different RGB-D feature fusion networks. And I explored the design of the cascade pipeline for tasks such as human pose estimation, human detection and human tracking.

The following abstract may give you a better understanding of my project.

## Abstract

Human pose estimation, human detection and human tracking are several most popular tasks in the field of computer vision. How to improve the accuracy and efficiency of these tasks has been the focus of researchers in recent years. With the continuous development of binocular cameras for various depth images, people have begun to shift their attention from traditional 2D RGB images to RGB-D images for these human tasks. Our work uses Lightweight-OpenPose as the basic backbone, aiming to explore two difficulties in RGB-D human tasks: human occlusions and system efficiency. 

Firstly, our work uses RGB-D images with depth information to improve the skeleton connections in the scene of human occlusions and overlapping, and explores several different RGB-D feature fusion networks while proposing a cross-fusion feature network. In order to evaluate the influence of depth information on the skeleton connections, our work also designs a skeleton connection index link-AP. On the self-made local dataset,
the proposed feature fusion network achieves significant improvements in AP in COCO format and link-AP metrics compared with the RGB model and other RGB-D models. Secondly, in order to improve the operation efficiency of the multi-task learning system, the human body detection and human tracking modules are connected in series after the human pose estimation module, and a human detection index using human joint points as input is designed: human-AP. Tested on the local dataset, the proposed cross-fusion feature network also has a better performance. In addition, this subject further verifies that the addition of depth information improves the recognition of human information in occlusion and overlapping scenes through detailed analysis on representative images.

## Intersted in My Work?

Due to time limitations, I only did some very preliminary explorations of RGB-D human pose estimation in my graduation project. If you want to know more about what I did in the project, please contact me and I can share with you my graduate thesis. If you agree with the ideas I put forward in this project or are interested in my project, please don't hesitate to contact me beccause I would love to discuss with distinguished explorers or learners like me in the field of computer vision and deep learning.
