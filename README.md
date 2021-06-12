# 1. Introduction
The goal of this project is to investigate if using style transfer as data augmentation to enlarge the dataset is beneficial and if so, if it more beneficial than enlarging your dataset via normal augmentations.
...

# 2. Style transfer
Style transfer can be used to compose images in the style of another image, or painting. This specific technique was first introduced by Gatys et al. [2]. It uses a style image (for example of a painting), and an input image, after which it tries to blend these two images together such that the input image is styled like the style image. For this project, three different style images were taken into account, which are shown below.
![all_styles](https://user-images.githubusercontent.com/61514183/120921464-14377300-c6c4-11eb-8544-5c1fc0fe08f1.jpg)
The first and second paintings are more abstract works, where the first painting shows a lot of rounded edges and lines and the second painting contains a lot of rectangles. The third painting is a more classic work of Van Gogh. The goal of this project is to compare the different abstract works to the classical work and see if one sort of painting proves to be more beneficial.

For the style transfer, we use a pretrained VGG19 model on Imagenet, as used by Gatys et al. Intermediate layers of this model are used to obtain the content and style features, as shown in below code block. Next, we introduce two losses that are combined and used for the style transfer: content loss and style loss.

```
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
 ```
The idea is that we have a content and style image that we both want to match, and transform a base input image such that it matches the content of the content image and the style of the style image. Hence, the total loss is simply the addition of the content and style loss. 

## Content loss
Initlially we pass the network the content and style image, from which it produces a stylized image. We can retrieve the style and content features from the layers that we defined above. We first do this for the original content image, after which we run the content image through the model and again retrieve the content features. The content loss here is defined as the euclidian distance between the content features of the input (content) image and the image that was generated. We can write this in code, where base_content represents the stylized image, and target represents the original content image. The goal here is to minimize the content loss. 

```
content_loss = tf.reduce_mean(tf.square(base_content - target))
```

## Style loss
To compute the style loss, we give the network the content and style image and compare the Gram matrices of the outputs. We try to minimize the mean squared distance between the features of the style and stylized image. Here, gram_target are the gram features from the style layers of the style image and gram_style are the gram featuers of the style layers of the stylized image.

```
gram_style = gram_matrix(base_style)
style_loss = tf.reduce_mean(tf.square(gram_style - gram_target))
```

## Dataset
As the goal of this project is to investigate if using style transfer can be used to enlarge a small dataset and thus increase performance, we had to choose a dataset that is relatively small and does not have a lot of images per class. Therefore, the Caltech 101 dataset was used for this project; originally it contains 101 classes, where most classes contain about fifty iimages [3]. The original idea was to use the full dataset for style transfer. However, as using style transfer on each image took roughly 2 to 4 minutes, only twelve classes were used in the end. Random images from the twelve different classes are shown below.
![classes](https://user-images.githubusercontent.com/61514183/121782674-165d6e00-cbab-11eb-9624-2515f6296192.png)

## Generating stylized images 
The total loss that was used was a combination of the earlier mentioned style and content loss. Then, gradient descent was run to transform the original into a stylized image that matches the style of the style image and content of the content image. We use the Adam optimizer to minimize the loss and run for 1000 iterations. The image with the lowest combined loss is saved into the data directory.

## Style transfer results
Examples of style transfer results for the classes headphone, water lilly, emu and wildcat are shown below. Here we see that the style transfer is done successfully in most cases. However, for style 2 we see that the original image is less recognizable.
![Screenshot from 2021-06-06 13-27-57](https://user-images.githubusercontent.com/61514183/120922729-105b1f00-c6cb-11eb-94fa-36b9345598b6.png)
![Screenshot from 2021-06-06 13-28-05](https://user-images.githubusercontent.com/61514183/120922730-118c4c00-c6cb-11eb-9830-7f39891ea8e0.png)

# 3. Models

# 4. Results
## Style transfer vs Original

|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Style 1 	|  0.914 	|  0.928 	|
|  Style 2 	|  0.890 	|  0.899 	|
|  Style 3	|   0.914	|   0.931	|
|  All styles	|  0.933 	|  0.916 	|


## Data augmentation vs Original

## Style transfer and data augmentation vs Original

# 5. Discussion

# 6. Conclusion

# References
[1] Zheng, X., Chalasani, T., Ghosal, K., Lutz, S., & Smolic, A. (2019). Stada: Style transfer as data augmentation. arXiv preprint arXiv:1909.01056.

[2] Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

[3] L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models from few training examples: an incremental Bayesian approach tested on 101 object categories. IEEE. CVPR 2004, Workshop on Generative-Model Based Vision. 2004
