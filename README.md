# GenGAN
SynthGAN is a GAN based model whose objective is to create synthetic collaborative datasets.

At present it consists of three components:
1. **Deep matrix factorization** that is used to perform dimension reduction and generate a new dataset of emebeddings.
2. **Generative Adversarial Network** is intended to be a fully modular component that can be built using just two build functions; one each for the generator and discriminator components.
3. **K-means Clustering** is used to group together similar user, item data points.

This project is inspired by the following papers:
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Deep matrix factorizations](https://arxiv.org/abs/2010.00380)
- [RSGAN](https://arxiv.org/abs/2303.01297)