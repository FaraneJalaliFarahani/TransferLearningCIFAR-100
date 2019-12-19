# TransferLeraningCIFAR-100 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1juyiqyGwVYLWl5MA2dJEQ13h67kPJ-IR#scrollTo=bX4dJAVWM4PM)

Methodology: Transfer Learning

Deep Learning Framework: Pytorch

Datasets: CIFAR-10 and CIFAR-100 (Download from the https://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. One of the very lucrative features about the CIFAR-10 dataset is that there is no need to download the images separately and load them into a directory. We are able to import it from the Torchvision Datasets packages. CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.

Inorder to classify 5 classes of CIFIAR-100(bicycle, bus, dolphin, mountain, table), the CNN classifier with 3 convolutional layers and 2 fully connected layers was trained on CIFAR-10. covcolutional layers of pre-trained model were freezed and the weights of the fully connected layers were fine-tuned by the backpropagation.

This model reached 73 percent accuracy on CIFAR-100 test images.
