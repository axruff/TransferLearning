
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Domain Adaptation and Transfer Learning
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->


##### Table of Contents
- [Domain Adaptation and Transfer Learning](#domain-adaptation-and-transfer-learning) 
  - [Surveys](#surveys)
  - [Discrepancy-based Approaches](#discrepancy-based-approaches)
  - [Adversarial-based Approaches](#adversarial-based-approaches)
    - [Generative Models](#generative-models) 
    - [Non-generative Models](#non-generative-models) 
  - [Reconstruction-based Approaches](#reconstruction-based-approaches)
  - [Synthetic Data](#synthetic-data)
  - [Style Transfer](#style-transfer) 
  - [Texture Synthesis](#texture-synthesis) 



Notations

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) Read and worked

![#ff0000](https://placehold.it/15/ff0000/000000?text=+) TODO


## Surveys 

[A Survey on Transfer Learning (2010)](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)

[Transfer learning for visual categorization: A survey (2015)](https://ieeexplore.ieee.org/document/6847217)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Domain Adaptation for Visual Applications: A Comprehensive Survey (2017)](https://arxiv.org/abs/1702.05374)

[Visual domain adaptation: A survey of recent advances (2015)](https://ieeexplore.ieee.org/document/7078994)

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+)
[Deep Visual Domain Adaptation: A Survey (2018)](https://arxiv.org/abs/1802.03601)

[A survey on heterogeneous transfer learning (2017)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0089-0)

[Transfer learning for cross-dataset recognition: A survey (2017)](https://arxiv.org/abs/1705.04396)

[Visual Domain Adaptation Challenge (2018)] (http://ai.bu.edu/visda-2017/)


## Discrepancy-based Approaches

*Description: fine-tuning the deep network with labeled or unlabeled target data to diminish the domain shift*

- **Class Criterion**: uses the class label information as a guide for transferring knowledge between different domains. When the labeled samples from the target domain are available in supervised DA, **soft label** and metric learning are always effective [118], [86], [53], [45], [79]. When such samples are unavailable, some other techniques can be adopted to substitute for class labeled data, such as **pseudo labels** [75], [139], [130],[98] and **attribute representation** [29], [118]. Usually a small
number of labeled samples from the target dataset is assumed to be available. 

- **Statistic Criterion**: aligns the statistical distribution shift between the source and target domains using some mechanisms. 

- **Architecture Criterion**: aims at improving the ability of learning more transferable features by adjusting the architectures of deep networks.

- **Geometric Criterion**: bridges the source and target domains according to their geometrical properties.

#### Class Criterion

Using **soft labels** rather than hard labels can preserve the relationships between classes across domains.

Humans can identify unseen classes given only a high-level description. For instance, when provided the description ”tall brown
animals with long necks”, we are able to recognize giraffes. To imitate the ability of humans, [64] introduced high-level **semantic attributes** per class.

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[Fine-grained recognition in the wild: A multi-task domain adaptation approach (2017)](https://arxiv.org/abs/1709.02476) [soft labels, semantic attributes]

[Deep transfer metric learning (2015)](https://ieeexplore.ieee.org/document/7298629)


Occasionally, when fine-tuning the network in unsupervised DA, a label of target data, which is called a pseudo label, can preliminarily be obtained based on the maximum posterior probability

[Mind the class weight bias: Weighted maximum mean discrepancy for unsupervised domain adaptation (2017)](https://arxiv.org/abs/1705.00609)

In [[98]](https://arxiv.org/abs/1702.08400), two different networks assign **pseudo-labels** to unlabeled samples, another network is trained by the samples to obtain target discriminative representations.

[Asymmetric tri-training for unsupervised domain adaptation (2017)](https://arxiv.org/abs/1702.08400)

[**[DTN]** Deep transfer network: Unsupervised domain adaptation (2015)](https://arxiv.org/abs/1503.00591)

#### Statistic Criterion

Although some discrepancy-based approaches search for pseudo labels, attribute labels or other substitutes to labeled
target data, more work focuses on learning **domain-invariant representations** via minimizing the domain distribution discrepancy in unsupervised DA.

**Maximum mean discrepancy** (MMD) is an effective metric for comparing the distributions between two datasets by a kernel two-sample test [3].

![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[**[DDC]** Deep domain confusion: Maximizing for domain invariance (2014)](https://arxiv.org/abs/1412.3474)

<img src="https://www.groundai.com/media/arxiv_projects/85020/x1.png.750x0_q75_crop.png" width="300">

Rather than using a single layer and linear MMD, Long et al. [[73]](https://arxiv.org/abs/1502.02791) proposed the deep adaptation network (DAN) that matches the shift
in marginal distributions across domains by adding multiple adaptation layers and exploring multiple kernels, assuming that the conditional distributions remain unchanged.

[**[DAN]** Learning transferable features with deep adaptation networks (2015)](https://arxiv.org/abs/1502.02791)

[**[JAN]** Deep transfer learning with joint adaptation networks (2016)](https://arxiv.org/abs/1605.06636)

[**[RTN]** Unsupervised domain adaptation with residual transfer networks (2016)](https://arxiv.org/abs/1602.04433)

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0925231218306684-gr6.jpg" width="400">

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Associative Domain Adaptation (2017)](https://arxiv.org/abs/1708.00938)

<img src="https://vision.in.tum.de/_media/spezial/bib/haeusser_iccv_17.png" width="400">

[Return of frustratingly easy domain adaptation (2015)](https://arxiv.org/abs/1511.05547)


#### Architectural Criterion

[Deeper, broader and artier domain generalization (2017)](https://arxiv.org/abs/1710.03077)

<img src="http://www.eecs.qmul.ac.uk/~dl307/img/project_img1.png" width="300">

#### Geometric Criterion

[**[Dlid]**: Deep learning for domain adaptation by interpolating between domains (2013)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.664.4509) [geometric criterion]

## Adversarial-based Approaches

*Description: using domain discriminators to encourage domain confusion through an adversarial objective*

### Generative Models

[Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[**[DANN]** Domain-Adversarial Training of Neural Networks (2015)](https://arxiv.org/abs/1505.07818) [github](https://github.com/fungtion/DANN)

<img src="https://camo.githubusercontent.com/5201a6af692fe44c22cc2dfda8e9db02fb0e0ffc/68747470733a2f2f73312e617831782e636f6d2f323031382f30312f31322f70384b5479442e6d642e6a7067" width="350">


[Improved techniques for training GANs (2016)](https://arxiv.org/abs/1606.03498) [github](https://github.com/openai/improved-gan)


[Domain Separation Networks (2016)](https://arxiv.org/abs/1608.06019)

<img src="https://i.pinimg.com/564x/de/50/fa/de50fac81074e16ca78114f78a379246.jpg" width="350">


![#c5ff15](https://placehold.it/15/c5ff15/000000?text=+) 
[**[PixelDA]** Unsupervised pixel-level domain adaptation with generative adversarial networks (2016)](https://arxiv.org/abs/1612.05424) 

***[Needs a lot of target images to successfully learn the generator]***

<img src="https://i.pinimg.com/564x/f8/52/1e/f8521e45415762465e5e01452a963a31.jpg" width="400">

[**[ADDA]** Adversarial discriminative domain adaptation (2017)](https://arxiv.org/abs/1702.05464)

<img src="https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/ADDA_1.jpg" width="400">

This weight-sharing constraint allows **CoGAN** to achieve a domain-invariant feature space
without correspondence supervision. A trained CoGAN can adapt the input noise vector to paired images that are from the
two distributions and share the labels. Therefore, the shared labels of synthetic target samples can be used to train the target
model.

[**[CoGAN]** Coupled generative adversarial networks (2016)](https://arxiv.org/abs/1606.07536)

[Pixel-level domain transfer (2016)](https://arxiv.org/pdf/1603.07442.pdf) [[github]](https://github.com/fxia22/PixelDTGAN)

<img src="https://pbs.twimg.com/media/CgKhQ2hWEAAE231.jpg:large" width="400">

Shrivastava et al. [104]() developed a method for **simulated+unsupervised (S+U)** learning
that uses a combined objective of minimizing an adversarial
loss and a self-regularization loss, where the goal is to improve
the realism of synthetic images using unlabeled real data

[Learning from Simulated and Unsupervised Images through Adversarial Training (2016)](https://arxiv.org/abs/1612.07828)

<img src="https://github.com/axruff/ML_papers/raw/master/images/123.png" width="300">

[Improved Adversarial Systems for 3D Object Generation and Reconstruction (2017)](https://arxiv.org/abs/1707.09557)


[Toward Multimodal Image-to-Image Translation (2018)](https://arxiv.org/abs/1711.11586)

<img src="https://junyanz.github.io/BicycleGAN/index_files/teaser.jpg" width="400"> 

[From Source to Target and Back: Symmetric Bi-Directional Adaptive GAN (2018)](http://openaccess.thecvf.com/content_cvpr_2018/html/Russo_From_Source_to_CVPR_2018_paper.html)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[SRDA: Generating Instance Segmentation Annotation Via Scanning, Reasoning And Domain Adaptation (2018)](https://arxiv.org/abs/1801.08839) ***[Geometry-guided GAN]***

[How good is my GAN? (2018)](https://arxiv.org/abs/1807.09499)

<img src="http://thoth.inrialpes.fr/research/ganeval/images/fig1.png" width="350">

[2019 - U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)

<img src="https://pythonawesome.com/content/images/2019/08/generator.png" width="350">


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
### Non-generative Models
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

## Reconstruction-based Approaches

*Description: using the data reconstruction as an auxiliary task to ensure feature invariance*


## Others

GAN Zoo
[[link]](https://github.com/hindupuravinash/the-gan-zoo)

<img src="https://github.com/hindupuravinash/the-gan-zoo/raw/master/The_GAN_Zoo.jpg" width="250">


Self-ensembling for visual domain adaptation
[[link]](https://arxiv.org/abs/1706.05208)

![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
[Context Encoders: Feature Learning by Inpainting (2016)](https://arxiv.org/abs/1604.07379) [[github]](https://github.com/pathak22/context-encoder)

<img src="https://i.pinimg.com/564x/57/be/d5/57bed585ea990b858314f919db5fc522.jpg" width="400">

[Compositional GAN: Learning Conditional Image Composition (2018)](https://arxiv.org/abs/1807.07560)

<img src="http://pbs.twimg.com/media/Di7lWdWXoAAhd6B.jpg" width="400">

GAN Dissection: Visualizing and Understanding Generative Adversarial Networks
[[link]](https://arxiv.org/abs/1811.10597v)
Interactive tool:
https://gandissect.csail.mit.edu/

Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning
[[link]](https://arxiv.org/abs/1704.03976)

<img src="https://www.researchgate.net/profile/Takeru_Miyato/publication/316098571/figure/fig2/AS:667791753498635@1536225369918/Demonstration-of-how-our-VAT-works-on-semi-supervised-learning-We-generated-8-labeled.png" width="300">

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.jpg" width="300">

[Pros and Cons of GAN Evaluation Measures (2018)](https://arxiv.org/abs/1802.03446)




<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Synthetic Data
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2013 - Simulation as an engine of physical scene understanding](https://www.pnas.org/content/110/45/18327)

[Playing for Data: Ground Truth from Computer Games](https://arxiv.org/abs/1608.02192)

<img src="https://github.com/axruff/ML_papers/raw/master/images/PlayingforData.png" width="300">


[2017 - On Pre-Trained Image Features and Synthetic Images for Deep Learning)](https://arxiv.org/abs/1710.10710)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-3-030-11009-3_42/MediaObjects/478770_1_En_42_Fig3_HTML.png" width="350">

[2017 - Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)

<img src="https://i.pinimg.com/564x/99/e9/65/99e9657382ef1e94e2acb958c7c1cf6c.jpg" width="350">


[2017 - Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization](https://arxiv.org/abs/1804.06516)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)
<img src="https://research.nvidia.com/sites/default/files/publications/cvpr-fig1_down4.png" width="350">


[2018 - Learning to Segment via Cut-and-Paste](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tal_Remez_Learning_to_Segment_ECCV_2018_paper.pdf)

<img src="https://cdn-images-1.medium.com/max/1600/0*b4CBigBlK_LyGU16" width="350">


<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Style Transfer
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2016 - Image Style Transfer Using Convolutional Neural Networks](https://ieeexplore.ieee.org/document/7780634)

<imf src="https://i-h1.pinimg.com/564x/8a/e4/97/8ae497d18a7c409c2da67833d5586461.jpg" width="250">


[2016 Perceptual losses for real-time style transfer and super-resolution](https://arxiv.org/abs/1603.08155?context=cs) [[github]](https://github.com/jcjohnson/fast-neural-style)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/RTNS.png" width="350">


[2017 - **[CycleGAN]** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

<img src="https://junyanz.github.io/CycleGAN/images/teaser.jpg" width="400">

[2016 - **[pix2pix]** Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)[[github]](https://phillipi.github.io/pix2pix/)


[2018 - **[DRIT]** Diverse Image-to-Image Translation via Disentangled Representations](https://arxiv.org/abs/1808.00948)[[github]](https://github.com/HsinYingLee/DRIT)
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)

<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-01246-5_3/MediaObjects/474172_1_En_3_Fig3_HTML.gif" width="350">

[2018 - A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)[[github]](https://github.com/NVlabs/stylegan)

<img src="https://github.com/NVlabs/stylegan/raw/master/stylegan-teaser.png" width="350">

[2018 - Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189v3) [[github]](https://github.com/luanfujun/deep-painterly-harmonization)

<img src="https://i.pinimg.com/564x/f6/fa/74/f6fa740395c1b99ecee2b71f46b16751.jpg" width="350">

<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->
# Texture Synthesis
<!--- ------------------------------------------------------------------------------- -->
<!--- =============================================================================== -->
<!--- ------------------------------------------------------------------------------- -->

[2015 - Texture Synthesis Using Convolutional Neural Networks](https://arxiv.org/abs/1505.07376) [[github]](https://mc.ai/tensorflow-implementation-of-paper-texture-synthesis-using-convolutional-neural-networks/)

<img src="https://dmitryulyanov.github.io/assets/online-neural-doodle/textures.png" width="300">

DeepTextures
http://bethgelab.org/deeptextures/

Textures database
https://www.textures.com/index.php

