# pixelCnn
The blog post associated with this repo can be found [here](https://ahernandez105.github.io/blogPosts/pixelCnn/pixelCnn.html).

## Training Black and White MNIST
<code>python train.py --pickle data/mnist.pkl --layers 5 --dev cuda --conv_class MaskedConv2dBinary --save_path output/model_binary.pt</code>

## Training Colored MNIST
Unfortunately, I do not have enough storage to keep the colored mnist dataset in this repo. you can obtain this dataset from the Berkeley Deep Unsupervised Learning [repo](https://github.com/rll/deepul) by unzipping deepul/homeworks/hw1/data/hw1_data.zip.

<code>python train.py --pickle data/mnist_colored.pkl --filters 120 --layers 8 --dev cuda --dist_size 4 --conv_class MaskedConv2dColor --nll_img_path output/color_nll.png --samples_img_path output/color_samples.png</code>