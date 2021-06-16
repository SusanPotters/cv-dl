By Meltem Coroz and Susan Potters

Link to code

# 1. Introduction
Recent work has suggested that ImageNet-trained CNNs are biased towards recognising textures, instead of shapes [1]. Moreover, they demonstrated that the same architecture that can learn a texture-based representation on ImageNet, is also able to learn a shape-based representation on a stylized version of ImageNet. Training on stylized images provides an additional robustness to image distortions, giving rise to the idea of using style transfer for data augmentation, which was also investigated by Zheng et al. [2]. 

The goal of this project is to investigate if using style transfer as data augmentation to enlarge the dataset is beneficial and if so, if it more beneficial than enlarging your dataset via normal data augmentations. In order to achive this, a selection of classes of the Caltech 101 dataset is used for style transfer to enlarge the dataset. Additionally, traditional data augmentation is used as another way to enlarge the dataset. Two models are trained on the resulting datasets, for which the results can be compared. Lastly, we investigate the additional benefit of using traditional data augmentations in combination with style transfer.


# 2. Style transfer
Style transfer can be used to compose images in the style of another image, or painting. This specific technique was first introduced by Gatys et al. [3]. It uses a style image (for example of a painting), and an input image, after which it tries to blend these two images together such that the input image is styled like the style image. For this project, three different style images were taken into account, which are shown below.
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
As the goal of this project is to investigate if using style transfer can be used to enlarge a small dataset and thus increase performance, we had to choose a dataset that is relatively small and does not have a lot of images per class. Therefore, the Caltech 101 dataset was used for this project; originally it contains 101 classes, where most classes contain about fifty iimages [4]. The original idea was to use the full dataset for style transfer. However, as using style transfer on each image took roughly 2 to 4 minutes, only twelve classes were used in the end. Random images from the twelve different classes are shown below. 
![classes](https://user-images.githubusercontent.com/61514183/121782674-165d6e00-cbab-11eb-9624-2515f6296192.png)

70 percent of the original dataset consisting of 12 classes was used for training, and thus was augmented. The remaining 30 percent was used for testing purposes.

## Generating stylized images 
The total loss that was used was a combination of the earlier mentioned style and content loss. Then, gradient descent was run to transform the original into a stylized image that matches the style of the style image and content of the content image. We use the Adam optimizer to minimize the loss and run for 1000 iterations. The image with the lowest combined loss is saved into the data directory.

## Style transfer results
Examples of style transfer results for the classes headphone, water lilly, emu and wildcat are shown below. Here we see that the style transfer is done successfully in most cases. However, for style 2 we see that the original image is less recognizable.
![Screenshot from 2021-06-06 13-27-57](https://user-images.githubusercontent.com/61514183/120922729-105b1f00-c6cb-11eb-94fa-36b9345598b6.png)
![Screenshot from 2021-06-06 13-28-05](https://user-images.githubusercontent.com/61514183/120922730-118c4c00-c6cb-11eb-9830-7f39891ea8e0.png)

# 3. Traditional augmentation
![augm](https://user-images.githubusercontent.com/61514183/122179074-7728c700-ce87-11eb-829d-d94b6b36a400.png)

# 4. Models
Two models were used for the analysis: VGG16 and VGG19. Both were pretrained on Imagenet and were further trained on the datasets that were created. The models were trained for 70 epochs, but early stopping was used if the model did not improve for too many epochs; the patience was set to 10. A learning rate of 0.0001 was used, in combination with the SGD optimizer. 

# 5. Results
The above mentioned models were each trained three times for the different datasets, after which results were averaged. This was done such that more significant conclusions can be extracted from the data.

## Style transfer vs Original
Below table shows the test accuracy results for both models. Results are shown for the original dataset, combinations of the original dataset and each style (so here the training dataset was multiplied by 2) and a combination of the original dataset and all three styles (so here the training dataset was multiplied by 4).

|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Style 1 	|  0.914 	|  0.928 	|
|  Style 2 	|  0.890 	|  0.899 	|
|  Style 3	|   0.914	|   0.931	|
|  All styles	|  0.933 	|  0.916 	|

![VGG16-and19-styles](https://user-images.githubusercontent.com/61514183/121886768-1fb71980-cd16-11eb-818b-e7e4e849c5cd.png)

We see that style 1 and 3 give improvements both models, whereas style 2 did not improve the test accuracy. When using all styles, there are improvements compared to only using the original training set. However, it should be noted that for VGG16 the performance increase is a little higher than for VGG19. For VGG16 the combined method yields higher results than for the seperate styles, while this is not the case for VGG19.

## Traditional data augmentation vs Original
Below table shows the test accuracies for VGG16 and V6619 for the original dataset and the augmented datasets, where the training dataset is doubled. Moreover, we investigated what the influence is of additional augmentations, where for each image in the training set five different augmentated images are generated. 

|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Data augmentation 1 	|  0.928 	|  0.922 	|
|  Data augmentation 5	|  0.933 	|  0.924 	|

![VGG16-and19-augmentations](https://user-images.githubusercontent.com/61514183/122212248-e9110880-cea7-11eb-9036-20bf5fe3d6a1.png)

Above table and figure show that there is an increase in performance when using traditional data augmentation, which are comparable to the performance increases when using style transfer for data augmentation. Using more data augmentations gives an additional performance boost, however the performance increase is relatively small. Moreover, the data augmentations have more benefit for the VGG16 network than the VGG19 network.

## Combined methods vs Original


|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Data augmentation 1 + Style 1 	|  0.937 	|  0.920 	|
|  Data augmentation 1 + Style 2 	|  0.878 	|  0.918 	|
|  Data augmentation 1 + Style 3 	|  0.939 	|  0.926 	|
|  Data augmentation 1 + All styles 	|  0.935 	|   	|
|  Data augmentation 5 + All styles 	|   	|   	|



# 6. Discussion
- use faster style transfer
- use different amounts of styling and see what it does
- do more runs so we can do t-tests and see if significant results

# 7. Conclusion

# References
[1] Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2018). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231.

[2] Zheng, X., Chalasani, T., Ghosal, K., Lutz, S., & Smolic, A. (2019). Stada: Style transfer as data augmentation. arXiv preprint arXiv:1909.01056.

[3] Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

[4] L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models from few training examples: an incremental Bayesian approach tested on 101 object categories. IEEE. CVPR 2004, Workshop on Generative-Model Based Vision. 2004
