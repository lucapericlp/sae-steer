# Steer

## Overview

Accepts a video as input, sampling frames upon which it does:
1. Generates segmentations via MobileSAM
2. Downsamples scene to 16x16 and filters out all segments < than 2x2 pixels post downsampling
3. Uses labelled features from the SAE of [./autolabel_pipeline/](./autolabel_pipeline/) to derive a feature vector for "still life with fruits"
3b. Prepare feature vector as 16x16 spatial map where all downsampled segments into 16x16 space carry the feature steering vector
4. Generate image from SDXL Turbo using a default styling prompt (i.e "isometric illustration") while perform feature steering on the 16x16 target block `down.2.1` during generation

## Acceptance Criteria
Successfully generates a video of "still life with fruits" on the basis of the segmentation mask derived from the video [./clips/waving.mp4](./clips/waving.mp4)

## Constraints
- The entire pipeline must run at a 1x RTF on an A10G GPU so ensure to benchmark the entire pipeline and optimize as necessary.
