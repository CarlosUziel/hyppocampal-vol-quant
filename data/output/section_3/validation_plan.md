# Validation plan of HyppoVolume.AI

This document details the plan to be followed in order to assess the production, real-world performance of the hyppocampal volume estimation algorithm.

## Intended Use

HyppoVolume.AI is meant to be used as an assistant to the radiologist by providing a fast and accurate estimation of hyppocampal volumes on a given brain MRI cropped image.

## Ground Truth Definition

In order to asses the performance of the algorithm, we need a robust ground truth that is reliable and represents the reality as much as possible. Given the known issue of inter-rater agreement, that is, the expected disagreement on segmentation labels by different radiologists, we propose the use of multiple domain-expert labelers to define the groud truth of each volume. The final ground truth would be determined by a majority voting system on each voxel or any suitable alternative, depending on the production conditions.

With this strategy, we guarantee that we are tuning the performance of our algorithm on the underlying reality of the disease, and not on the potential biases originated from the labeler.

An additional consideration is obtaining a representative sample of the population we want to address with our algorithm. Given that the algorithm focuses on Alzheimer's progression, it would make sense to focus on the older spectrum of the population. Whether it makes sense to include a balanced proportion of males and females, as well as different races, would depend on the incidence rate of this disease on these population subgroups.

## Performance Assesment Definition

We use multiple performance metrics to assess the goodness of our segmentation algorithm, including Dice's coefficient, Jaccard's coefficient, sensitiviy and specificity.

To assess whether a certain hyppocampal volume has changed size significantly, we need to take into account previous scans, if any, as well as the population trends. Population trend depends on age and sex, as it can be seen in the following figure for the case of women:

<img src="../../../images/nomogram_fem_right.svg" width=300>

This can be used to assess in which percentile the patient under consideration is, and whether it matches the expected distribution.

This population information can also be used as a ground truth reference. For example, we should expect our estimated volumes on healthy patients to match the previously shown trend.

## Input Data Definition

Our algorithm has been designed to work on T1-weighted MPRAGE MRI images cropped around the hyppocampal region of the brain. This means that it won't work on full brain MRI images or on different MRI modalities. Therefore, it must be guaranteed in the clinical setting that this is the kind of input received by the algorithm.
