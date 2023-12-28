# reproducible high-fidelity-generative-compression

we slight modified the project so that it can run on newer cuda verion(11.7).

## Usage
*make sure to have cuda/11.7,and change the module:
```bash
module load cuda/11.7
```
* Install Pytorch nightly and dependencies from [https://pytorch.org/](https://pytorch.org/). Then install other requirements:

```bash
pip install -r requirements.txt
```

* Clone this repository, `cd` in:

```bash
git clone git@github.com:littlepenguin89106/compression_final.git
cd high-fidelity-generative-compression
```


### Training

* Download a large (> 100,000) dataset of diverse color images. We found that using 1-2 training divisions of the [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset was able to produce satisfactory results on arbitrary images. Add the dataset path under the `DatasetPaths` class in `default_config.py`.

* For best results, as described in the paper, train an initial base model using the rate-distortion loss only, together with the hyperprior model, e.g. to target low bitrates:

```bash
# Train initial autoencoding model
python3 train.py --model_type compression --regime low --n_steps 1e6
```

* Then use the checkpoint of the trained base model to 'warmstart' the GAN architecture. Please see the [user's guide](src/README.md) for more detailed instructions.

```bash
# Train using full generator-discriminator loss
python3 train.py --model_type compression_gan --regime low --n_steps 1e6 --warmstart -ckpt path/to/base/checkpoint
```

### Compression

* `compress.py` will compress generic images using some specified model. This performs a forward pass through the model to yield the quantized latent representation, which is losslessly compressed using a vectorized ANS entropy coder and saved to disk in binary format. As the model architecture is fully convolutional, this will work with images of arbitrary size/resolution (subject to memory constraints).

```bash
python3 compress.py -i path/to/image/dir -ckpt path/to/trained/model --reconstruct
```
The compressed format can be transmitted and decoded using the routines in `compress.py`. The [Colab demo](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb) illustrates the decoding process.

### WGAN-GP, WGAN-div

Checkout to `wgan` branch to use wgan implementation.
