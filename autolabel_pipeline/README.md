# AutoLabel SAE Features
In the blog post [./blog_post.md](./blog_post.md), the authors describe how they went about automatically labelling the 10,240 features which can be summarized as follows:
1. For each SAE feature, perform a single step unconditional SDXL-Turbo generation whose target block `down.2.1` gets feature steered with the candidate feature. Repeat this 3x times to get 3 images per feature.
2. Get the 3 images per feature and pass them to `Qwen2.5-VL-7B-Instruct` to generate a class label for the feature.
3. Repeat for all 10,240 features to get a mapping between feature index and class label.

You can find the code and the links for the weights for the SAE in https://github.com/goodfire-ai/sdxl-turbo-interpretability
