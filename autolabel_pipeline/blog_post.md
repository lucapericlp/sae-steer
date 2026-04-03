Research

We're launching [*Paint With Ember*](https://paint.goodfire.ai/) — a tool for generating and editing images by directly manipulating the neural activations of AI models. We're also [open-sourcing the SAE model](https://github.com/goodfire-ai/sdxl-turbo-interpretability/tree/main) that powers the app, and sharing our findings on diffusion models and the features they learn.

\* Goodfire
† Core contributor
Correspondence to dan@goodfire.ai![Paint With Ember launch header image](https://flash-blog-post.s3.us-west-2.amazonaws.com/hero.jpg)

Paint With Ember launch header image

Mechanistic interpretability techniques unlock new and powerful ways of interacting with generative models. By reverse engineering an image model to understand the visual features it's learned, we can then use these features to edit and create images in novel ways.

**[*Paint With Ember*](https://paint.goodfire.ai/) is a tool that replaces the traditional interface between creators and image models — the familiar prompt box — with a canvas that plugs directly into the "brain" of the model.** While it is still possible to guide the model via text prompts (for example, to specify a style), the canvas offers a 2D interface for expressing creative intent with familiar tools like painting and dragging. These actions correspond to manipulating the model's internal activations at specific spatial regions of the image being generated. Here are a few examples of the app in action:

![Painting with concepts](https://flash-blog-post.s3.us-west-2.amazonaws.com/example-gifs/paint-new-objects.gif)

Painting with concepts: Add new objects to the scene by painting with different factors.

This application complements Goodfire's work across other model architectures and modalities, such as natural language reasoning models \[1\] **Under the hood of a reasoning model** [\[HTML\]](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model)

D. Hazra, M. Loeffler, M. Cubuktepe, L. Avagyan, L. Gorton, M. Bissell, O. Lewis, T. McGrath, and D. Balsam, Goodfire Research, 2025. and genomic foundation models \[2\] **Interpreting Evo 2: Arc Institute's Next-Generation Genomic Foundation Model** [\[HTML\]](https://www.goodfire.ai/blog/interpreting-evo-2)

M. Deng, D. Balsam, L. Gorton, N. Wang, N. Nguyen, E. Ho, and T. McGrath, Goodfire Research, Feb. 20, 2025.. We believe in the importance of combining interpretability research with intuitive interfaces for understanding and wielding powerful AI systems. This applies across all domains — whether in the pursuit of making novel scientific discoveries, improving safety and reliability for enterprise applications, or unlocking new forms of creative expression. We also believe interpretability will unlock entirely new form factors for working with AI, and that inventing new unsupervised techniques for understanding model latents is essential for scaling interpretability to superintelligent systems. Paint With Ember is a fun and accessible synthesis of these goals.

We invite you to explore the public version of the application [here](https://paint.goodfire.ai/), and make use of our open-source SAE interpreter model [here](https://github.com/goodfire-ai/sdxl-turbo-interpretability/tree/main).

## Zoom in, Zoom Out: Features and Factors

Paint With Ember uses [Stable Diffusion XL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) as its underlying image generation model. SDXL-Turbo is a 3.5B parameter latent diffusion model that can produce images in as few as 1-4 timesteps, allowing for near real-time generation.

We focused on interpreting a specific layer of SDXL-Turbo — block `down.2.1` — where the model determines the overall composition of the image it's generating. At this layer, the model breaks the image into 16x16 patches. Patches are analogous to tokens in a language model; each of the 256 patches is represented by a vector in the model's latent space, and patches interact with each other and the prompt through self- and cross-attention.

Block `down.2.1` occurs relatively early in the model's UNet and is generally known to be the location at which SDXL-Turbo chooses the primary elements of the foreground and background of the image. The layer was investigated in detail by Surkov, et. al \[3\] **Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders** [\[link\]](https://arxiv.org/abs/2410.22366)

Viacheslav Surkov, Chris Wendler, Mikhail Terekhov, Justin Deschenaux, Robert West, and Caglar Gulcehre, 2024. arXiv preprint arXiv:2410.22366. who trained sparse autoencoder (SAE) models across various layers of SDXL-Turbo.

To decompose each patch's latent vector into a set of interpretable features, we trained a BatchTopK SAE \[4\] **BatchTopK Sparse Autoencoders** [\[link\]](https://arxiv.org/abs/2412.06410)

Bart Bussmann, Patrick Leask, and Neel Nanda, 2024. arXiv preprint arXiv:2412.06410.. This SAE lets us ask: "what are the top features the model is thinking about in each patch?" With a BatchTopK SAE, the model allocates these features non-uniformly across patches, allocating some patches with more features than others. You can explore the features in each patch of the example image below, using the toggle to visualize the count of features in each patch.

<iframe src="https://flash-blog-post.s3.us-west-2.amazonaws.com/feature_patches.html" height="310px !important"></iframe>

BatchTopK SAE features for a single-timestep output from SDXL-Turbo, using the prompt:
`two golden retrievers looking at each other in a garden full of roses`

The SAE makes the model's latents more legible by converting them into recognizable visual features, but these features are too fine-grained for most of the ways we'd like to interact with an image. There are hundreds of active SAE features across an image, and it would be nice to have access to higher-level abstractions that bundle related features together — for example, combining dozens of features into a single "dog" unit. These higher-level units enable useful actions that require adjusting or shifting many features at a time, such as moving the flowers to the left, making the grass taller, or adding a dog in an entirely new image.

We create these higher-level units by performing non-negative matrix factorization (NMF) on the tensor that represents the feature content of each image patch. This approach was inspired by Olah, et. al. \[5\] **The Building Blocks of Interpretability** [\[HTML\]](https://distill.pub/2018/building-blocks)

Olah, C., Satyanarayan, A., Johnson, I., Carter, S., Schubert, L., Ye, K., and Mordvintsev, A., 2018., who used NMF to create neuron groups as a strategy for making interfaces more "human-scale":

> If we want to make useful interfaces into neural networks, it isn't enough to make things meaningful. We need to make them human scale, rather than overwhelming dumps of information. [(Distill, 2018)](https://distill.pub/2018/building-blocks/)

NMF lets us decompose the image into `N` components we'll refer to as *factors* (also called *concepts* in the Paint With Ember app). Each factor is a weighted sum of SAE features. Below, you can explore how NMF decomposes our example image across varying factor counts.

<iframe src="https://flash-blog-post.s3.us-west-2.amazonaws.com/factor_counts_new.html" width="100%"></iframe>

## The Neural Canvas

To sum up: the primary conceptual units in the Paint With Ember application are **factors**, which we call *concepts* in the app, and **features**, which are the sub-units of factors. Factors are weighted sums of features, and can be thought of as clusters of features that represent higher-level visual elements. Both features and factors are found in an unsupervised manner, although features are global properties of the model learned by the SAE whereas factors are generated per image using NMF.

All of the core actions you can take in the application involve manipulating either the spatial positioning of factor strengths across the image, or manipulating the weighted feature vector that defines each factor.

**Spatial edits like painting, dragging, and erasing concepts** correspond to setting, moving, or ablating factor strengths across the `16 x 16 x           n_factors` tensor that represents which factors are active in each patch and how strongly they are active.

**Increasing or decreasing the strength of a factor across the entire image** corresponds to scaling the given factor index across the entire 3rd dimension of this tensor. (You could also adjust the strength of the factor in particular regions by only scaling values in those patches.)

**Adjusting the semantic content of a factor** corresponds to changing the weights of the features that define each factor in an `n_factors x n_features` matrix produced by NMF (sometimes called the *basis matrix*).

Image generation can also be influenced via a text prompt. The current public application uses the prompt exclusively to provide style guidance, although in earlier versions of the application we experimented with combining open-ended prompting with steering via the spatial canvas <sup>1The prompt and the injected spatial activations can interact in unintuitive ways, particularly when they offer conflicting signals for what image to generate. Based on user testing, we found that limiting the prompt to style guidance provided a more enjoyable experience.</sup>. Note that image generation works well even when no prompt is provided, i.e. when the only input guiding the model's generation is the injected neural activations from the painted canvas. We anticipate that training SAEs on later layers of SDXL-Turbo could enable style guidance without using the prompt at all; this is an area of interest for future work.

## Features

*Paint With Ember* uses a BatchTopK SAE \[4\] **BatchTopK Sparse Autoencoders** [\[link\]](https://arxiv.org/abs/2412.06410)

Bart Bussmann, Patrick Leask, and Neel Nanda, 2024. arXiv preprint arXiv:2412.06410. with `k=10` and an expansion factor of `8`. These hyperparameters showed the best quantitative and qualitative results when sweeping values of 10-48 for k and 4-16 for the expansion factor.

We used DataMapPlot to create an interactive UMAP \[6\] **Umap: Uniform manifold approximation and projection for dimension reduction** [\[link\]](https://arxiv.org/pdf/1802.03426)

McInnes, L., Healy, J., and Melville, J., 2018. arXiv preprint arXiv:1802.03426. visualization of the 10,240 SAE features, shown below. Each feature is visualized using an icon produced by a single-step generation from SDXL-Turbo, conditioned on an empty prompt and steered towards the given feature vector. We then used an automated labeling pipeline \[7\] **Language models can explain neurons in language models** [\[link\]](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)

Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, William Saunders, 2023. to label each feature by providing Claude 3.7 Sonnet with a set of three feature visualizations from distinct random seeds. Note that the interactive UMAP supports keyword search across these labels.

Many interpretable clusters of features can be explored, including regions related to [architecture](#), [hats](#), [text](#), [facial features](#), [vehicles/methods of transportation](#), etc. We've also highlighted a handful of features in the dropdown below <sup>2Note that the titles in the dropdown are our annotations and may differ from the auto-interpreted labels in the UMAP.</sup>. Clicking on a feature opens Paint With Ember with the feature pre-loaded for painting.

<iframe src="https://flash-blog-post.s3.us-west-2.amazonaws.com/umap_embedded.html" width="100%" height="800px"></iframe>

For a full-screen version of the UMAP, click [here](https://paint.goodfire.ai/umap.html).

A few notable categories of features include:

**Features for Specific Entities**

The SAE learned several features for surprisingly specific entities and concepts, including particular works of art (e.g. Munch's ["The Scream"](#)), logos (e.g. the [Bitcoin ₿](#)), buildings/locations (e.g. the [Oval Office](#)), and characters (e.g. [Buzz Lightyear](#)).

We did not expect to find such granular features at a layer this early in the network. Instead, we expected the model to operate with general features as it organizes the high-level image contents, which it would then refine in later layers. For example, when the model generates images of specific people mentioned in a prompt, we notice it uses generic "human face" features in layer `down.2.1`, and uses later cross-attention layers to add the details that turn these features into particular people. Evidently, certain concepts are distinctive enough or appear frequently enough in the training data for SDXL-Turbo and/or our SAE to warrant dedicated features.

**Features Representing Spatial Positioning Information**

We found several features that appear to identify directions in latent space the model uses to encode positional information. For example, we found features that were exclusively active in the four corners of images [(feature 770)](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=770) or along the top border [(feature 6214)](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=6214). These findings agree with other research demonstrating evidence that [SDXL-Turbo learns its own positional grid](https://x.com/rgilman33/status/1920785912474067341) since the model does not use positional embeddings.

Note that the visualizations and labels for these spatial features are often uninterpretable or misleading; the abstract notion of spatial boundaries cannot easily be represented visually, highlighting a shortcoming of feature visualization <sup>3Note that the visualizations and labels for these spatial features are often uninterpretable or misleading; the abstract notion of spatial boundaries cannot easily be represented visually, highlighting a shortcoming of feature visualization.</sup>.

**Features for Actions and Interactions Between Image Subjects**

While most features represent physical objects and entities, we also found features that seem to correspond to actions and properties such as "opening one's mouth" [(feature 7446)](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=7446) and "putting hands on one's face in distress" [(feature 4045)](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=4045), as well as interactions between subjects such as "hugging" [(feature 4071)](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=4071) and "squaring up to fight" [(feature 9820)](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=9820).

These features can be visualized in isolation — for example, to paint a disembodied open mouth — but are most interesting when used in conjunction with other features. "Interaction features" influence how the model interprets subjects and their relationships to one another. For example, adding a region of the "fighting" feature between two animals causes them to face each other and bear their teeth.

Most interestingly, these features demonstrate that SDXL-Turbo has learned how to generalize specific actions across different subjects and scenes. Applying the "hands on face in distress" feature to various animals does not cause them to develop human hands or facial features; instead, the feature adapts to its context, showing the specific animal's facial features becoming agitated and the analogous limb being used to touch its face (e.g. monkey paws or Minecraft block arms).

<iframe src="https://flash-blog-post.s3.us-west-2.amazonaws.com/features/gif-gallery.html" width="100%" height="250px"></iframe>

## Factors

While features capture granular details, factors represent higher-level abstractions of features, grouped contextually per image. Factors are dynamically generated by clustering related features based on their co-occurrence and spatial layouts in a particular image.

The feature content of an image is represented by a matrix of SAE feature weights across all 256 patches. Using non-negative matrix factorization (NMF), we can factorize this matrix into two non-negative matrices whose product approximates the original matrix. Doing so produces clusters of features, providing a useful set of higher-level concepts in an unsupervised manner.

Empirically, we find that using `3<=N<=8` typically decomposes an image into the most intuitive primary elements one might want to paint with <sup>4NMF will often dedicate entire factors for the specific "spatial information" features described in the <a href="#features-section">Features</a> section, leading to factors that occur in stripes along the borders of the image. Since we'd like factors to represent conceptual elements, we ablate these factors prior to fitting the NMF model for a given image. Although we ablate the features when <em>fitting</em> the NMF model, we include the features when <em>transforming the matrix</em> with the model after it has been fit. This approach did not noticeably affect the output image.</sup>. Many factors are dominated by a single prominent SAE feature, while others aggregate a more entropic blend of features.

In the Paint With Ember app, NMF is used to allow users to generate scenes which are factorized into concepts that the user can select for painting. After computing the SAE features and performing NMF, we perform auto-interpretation with a multimodal LLM to assign a label to each factor in the scene. In the example below, for `n_components=3`, auto-interpretation assigns the labels `background trees`, `galloping horse`, and `grassy field` to the factors visualized beneath the original image.

![Factors](https://flash-blog-post.s3.us-west-2.amazonaws.com/horse-autointerp.png)

In addition to providing a human-scale unit of abstraction, NMF offers an unsupervised way of finding distinct blends of features that represent concepts of interest. For example, while no single feature learned by our SAE seems to entirely represent a full cat head, the unique blend of features shown in the leftside image below does. Simply using the greatest activating feature in this "cat head" factor produces the creature shown on the right, indicating that factors can capture important interactions between sets of many SAE features rather than simply selecting the most prominent SAE feature to represent a region of the image <sup>5We might prefer that the SAE learns a single "cat head" feature, but in practice we observe the model learning granular features for various styles and sub-characteristics of cats such as cat eyes, cat face closeup images, fur around a cat face, etc. (see feature IDs <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=4443">4443</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=10155">10155</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=2548">2548</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=4419">4419</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=6517">6517</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=3149">3149</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=2824">2824</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=10151">10151</a>, <a href="https://paint.goodfire.ai/umap.html?embedded=true&amp;jumpTo=8906">8906</a>). We are interested in other strategies for grouping features learned through sparse dictionary learning methods (e.g. Anthropic's usage of <a href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html#appendix-dupe-features">supernodes</a> when constructing attribution graphs for cross-layer transcoders) as well as hierarchical methods of representation learning that address the issue of feature splitting (e.g. <a href="https://arxiv.org/abs/2503.17547">Matroyshka SAEs</a> and <a href="https://hkamath.me/blog/2024/rqae/">Residual Quantization Autoencoders</a>).</sup>.

![Factors](https://flash-blog-post.s3.us-west-2.amazonaws.com/factor-purity.png)

Left: the result of painting with a "cat head" factor composed of a distinct blend of features found using NMF. Right: the result of painting with only the top feature for this factor.

## Why We Built This

Goodfire's mission is to produce the research and tools that will help humanity understand and intentionally design the next frontier of AI systems. Paint with Ember connects to our broader goals by demonstrating how interpretability techniques create opportunities for interacting with models in ways that would otherwise be impossible if we capitulate to them as black boxes.

Image models have long served as ideal systems to test and expand our ability to interpret AI, since visual outputs are easier to comprehend and work with than other modalities <sup>6In particular, they do not require specific domain knowledge, operate on continuous pixel values rather than discrete tokens, and provide outputs that can be easily and quickly inspected.</sup>. However, our interpreter model, analysis agent, and white-box model interface techniques are not unique to image models and scale nicely to any neural network architecture. For example, **extrapolating the functionality of a tool like Paint With Ember to scientific models, one can imagine identifying key features and concepts across genomes instead of images, or "painting" specifically designed molecules using a generative materials science model.** Meanwhile, lessons learned from building the interfaces human operators need to interact with models at varying levels of abstraction is important regardless of the domain. We are excited for a future in which tools like Paint With Ember exist to help us reliably understand and interact with models of all kinds.

### Footnotes

1. The prompt and the injected spatial activations can interact in unintuitive ways, particularly when they offer conflicting signals for what image to generate. Based on user testing, we found that limiting the prompt to style guidance provided a more enjoyable experience.
2. Note that the titles in the dropdown are our annotations and may differ from the auto-interpreted labels in the UMAP.
3. Note that the visualizations and labels for these spatial features are often uninterpretable or misleading; the abstract notion of spatial boundaries cannot easily be represented visually, highlighting a shortcoming of feature visualization.
4. NMF will often dedicate entire factors for the specific "spatial information" features described in the [Features](#features-section) section, leading to factors that occur in stripes along the borders of the image. Since we'd like factors to represent conceptual elements, we ablate these factors prior to fitting the NMF model for a given image. Although we ablate the features when *fitting* the NMF model, we include the features when *transforming the matrix* with the model after it has been fit. This approach did not noticeably affect the output image.
5. We might prefer that the SAE learns a single "cat head" feature, but in practice we observe the model learning granular features for various styles and sub-characteristics of cats such as cat eyes, cat face closeup images, fur around a cat face, etc. (see feature IDs [4443](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=4443), [10155](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=10155), [2548](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=2548), [4419](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=4419), [6517](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=6517), [3149](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=3149), [2824](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=2824), [10151](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=10151), [8906](https://paint.goodfire.ai/umap.html?embedded=true&jumpTo=8906)). We are interested in other strategies for grouping features learned through sparse dictionary learning methods (e.g. Anthropic's usage of [supernodes](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#appendix-dupe-features) when constructing attribution graphs for cross-layer transcoders) as well as hierarchical methods of representation learning that address the issue of feature splitting (e.g. [Matroyshka SAEs](https://arxiv.org/abs/2503.17547) and [Residual Quantization Autoencoders](https://hkamath.me/blog/2024/rqae/)).
6. In particular, they do not require specific domain knowledge, operate on continuous pixel values rather than discrete tokens, and provide outputs that can be easily and quickly inspected.

### References

1. **Under the hood of a reasoning model** [\[HTML\]](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model)
	D. Hazra, M. Loeffler, M. Cubuktepe, L. Avagyan, L. Gorton, M. Bissell, O. Lewis, T. McGrath, and D. Balsam, Goodfire Research, 2025.
2. **Interpreting Evo 2: Arc Institute's Next-Generation Genomic Foundation Model** [\[HTML\]](https://www.goodfire.ai/blog/interpreting-evo-2)
	M. Deng, D. Balsam, L. Gorton, N. Wang, N. Nguyen, E. Ho, and T. McGrath, Goodfire Research, Feb. 20, 2025.
3. **Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders** [\[link\]](https://arxiv.org/abs/2410.22366)
	Viacheslav Surkov, Chris Wendler, Mikhail Terekhov, Justin Deschenaux, Robert West, and Caglar Gulcehre, 2024. arXiv preprint arXiv:2410.22366.
4. **BatchTopK Sparse Autoencoders** [\[link\]](https://arxiv.org/abs/2412.06410)
	Bart Bussmann, Patrick Leask, and Neel Nanda, 2024. arXiv preprint arXiv:2412.06410.
5. **The Building Blocks of Interpretability** [\[HTML\]](https://distill.pub/2018/building-blocks)
	Olah, C., Satyanarayan, A., Johnson, I., Carter, S., Schubert, L., Ye, K., and Mordvintsev, A., 2018.
6. **Umap: Uniform manifold approximation and projection for dimension reduction** [\[link\]](https://arxiv.org/pdf/1802.03426)
	McInnes, L., Healy, J., and Melville, J., 2018. arXiv preprint arXiv:1802.03426.
7. **Language models can explain neurons in language models** [\[link\]](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
	Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, William Saunders, 2023.

### Citation

Cammarata, et al., "Painting with concepts using diffusion model latents", Goodfire Research, 2025.

```
@article{cammarata2025painting,
  author = {Cammarata, Nick and Bissell, Mark and Nguyen, Nam and Deng, Myra and Ho, Eric and Gorton, Liv and Loeffler, Max and Balsam, Daniel},
  title = {Painting with concepts using diffusion model latents},
  journal = {Goodfire},
  year = {2025},
  note = {https://paint.goodfire.ai/}
}
```
