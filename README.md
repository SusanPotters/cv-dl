By Meltem Coroz and Susan Potters

[Link to code](https://github.com/SusanPotters/CV-DL-style-transfer)

# 1. Introduction
Recent work has suggested that ImageNet-trained CNNs are biased towards recognising textures, instead of shapes [1]. Moreover, they demonstrated that the same architecture that can learn a texture-based representation on ImageNet, is also able to learn a shape-based representation on a stylized version of ImageNet. Training on stylized images provides an additional robustness to image distortions, giving rise to the idea of using style transfer for data augmentation, which was also investigated by Zheng et al. [2]. 

The goal of this project is to investigate if using style transfer as data augmentation to enlarge the dataset is beneficial and if so, if it is more beneficial than enlarging your dataset via normal data augmentations. In order to achieve this, a selection of classes of the Caltech 101 dataset is used for style transfer to enlarge the dataset. Additionally, traditional data augmentation is used as another way to enlarge the dataset. Two models are trained on the resulting datasets, for which the results can be compared. Lastly, we investigate the additional benefit of using traditional data augmentations in combination with style transfer.


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

## Content loss
Initlially we pass the network the content and style image, from which it produces a stylized image. We can retrieve the style and content features from the layers that we defined above. We first do this for the original content image, after which we run the content image through the model and again retrieve the content features of the stylized image. The content loss here is defined as the Euclidian distance between the content features of the input (content) image and the stylized image that was generated. We can write this in code, where base_content represents the stylized image, and target represents the original content image. The goal here is to minimize the content loss. 

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
As the goal of this project is to investigate if style transfer can be used to enlarge a small dataset and thus increase performance, we had to choose a dataset that is relatively small and does not have a lot of images per class. Therefore, the Caltech 101 dataset was used for this project; originally it contains 101 classes, where most classes contain about fifty images [4]. The original idea was to use the full dataset for style transfer. However, as using style transfer on each image took roughly 2 to 4 minutes, only twelve classes were used in the end. Random images from the twelve different classes are shown below. 
![classes](https://user-images.githubusercontent.com/61514183/121782674-165d6e00-cbab-11eb-9624-2515f6296192.png)

70 percent of the original dataset consisting of 12 classes was used for training, and thus was augmented. The remaining 30 percent was used for testing purposes.

## Generating stylized images 
The total loss that was used was a combination of the earlier mentioned style and content loss. Then, gradient descent was run to transform the original into a stylized image that matches the style of the style image and content of the content image. We use the Adam optimizer to minimize the loss and run for 1000 iterations. The image with the lowest combined loss is saved into the data directory.

## Style transfer results
Examples of style transfer results for the classes headphone, water lilly, emu and wildcat are shown below. Here we see that the style transfer is done successfully in most cases. However, for style 2 we see that the original image is less recognizable.
![Screenshot from 2021-06-06 13-27-57](https://user-images.githubusercontent.com/61514183/120922729-105b1f00-c6cb-11eb-94fa-36b9345598b6.png)
![Screenshot from 2021-06-06 13-28-05](https://user-images.githubusercontent.com/61514183/120922730-118c4c00-c6cb-11eb-9830-7f39891ea8e0.png)

# 3. Traditional augmentation
Additionally, traditionl data augmentation was used. Below examples for the classes elephant, chandelier, pizza and cougar body are shown. The augmentations consisted of combinations of random rotations, horizontal and vertical shifts, zooming, horizontal flips, shear and changes in brightness.

```
datagen = ImageDataGenerator(
        rotation_range=10, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2],
        shear_range=0.2,
        fill_mode = 'nearest',
        channel_shift_range=20.0)
```

![augm](https://user-images.githubusercontent.com/61514183/122179074-7728c700-ce87-11eb-829d-d94b6b36a400.png)

# 4. Models
Two models were used for the analysis: VGG16 and VGG19. Both were pretrained on Imagenet and were further trained on the datasets that were created. The models were trained for 70 epochs, but early stopping was used if the model did not improve for too many epochs; the patience was set to 10. A learning rate of 0.0001 was used, in combination with the SGD optimizer. 

# 5. Results
The above mentioned models were each trained three times for the different datasets, after which results were averaged. After doing the first tests it was noticed that there were only slight increases or decreases, so three runs were done for each combination such that more significant conclusions could be extracted from the data.

## Style transfer vs Original
Below table shows the test accuracy results for both models. Results are shown for the original dataset, combinations of the original dataset and each style (so the amount of training samples is 2x more than the original) and a combination of the original dataset and all three styles (amount of training samples is 4x more than the original).

|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Style 1 	|  0.914 	|  0.928 	|
|  Style 2 	|  0.890 	|  0.899 	|
|  Style 3	|   0.914	|   0.931	|
|  All styles	|  0.933 	|  0.916 	|

![VGG16-and19-styles](https://user-images.githubusercontent.com/61514183/121886768-1fb71980-cd16-11eb-818b-e7e4e849c5cd.png)

We see that style 1 and 3 give improvements for both models, whereas style 2 did not improve the test accuracy. When using all styles, there are improvements compared to only using the original training set. However, it should be noted that for VGG16 the performance increase is a little higher than for VGG19. For VGG16, the combined method yields higher results than for the seperate styles, while this is not the case for VGG19.

## Traditional data augmentation vs Original
Below table shows the test accuracies for VGG16 and VGG19 for the original dataset and the augmented datasets, where the training dataset is doubled. Moreover, we investigated what the influence is of additional augmentations, where for each image in the training set five different augmentated images are generated. 

|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Data augmentation 1 	|  0.928 	|  0.922 	|
|  Data augmentation 5	|  0.933 	|  0.924 	|

![VGG16-and19-augmentations](https://user-images.githubusercontent.com/61514183/122212248-e9110880-cea7-11eb-9036-20bf5fe3d6a1.png)

The figure and table above show that there is an increase in performance when using traditional data augmentation, which is comparable to the performance increases when using style transfer for data augmentation. Using more data augmentations gives an additional performance boost, however the performance increase is relatively small. Moreover, the data augmentations are more beneficial for the VGG16 network than the VGG19 network.

## Combined methods vs Original
Below table and figure show results for both methods combined.

|   	|  VGG16 - average test accuracy	|   VGG19 - average test accuracy	|
|---	|---	|---	|
|  Original 	|  0.901 	|  0.905 	|
|  Data augmentation 1 + Style 1 	|  0.937 	|  0.920 	|
|  Data augmentation 1 + Style 2 	|  0.878 	|  0.918 	|
|  Data augmentation 1 + Style 3 	|  0.939 	|  0.926 	|
|  Data augmentation 1 + All styles 	|  0.935 	|  0.939 	|
|  Data augmentation 5 + All styles 	|   0.958	|  0.941 	|

![VGG16-and19-augmentations-styles](https://user-images.githubusercontent.com/61514183/122279955-ec78b400-cee8-11eb-869c-85e209adeab8.png)

The results show that the combination of normal augmentations and style transfer are the most beneficial for the VGG16 network for one augmentation and styles 1 and 3. It can also be seen that augmentation and style 2 combination is generally the worst, but still better than the original for VGG19 network. The biggest performance gain is achieved for VGG16 by using 5 augmentations and all styles, which is a gain of 6.3%.
# 6. Discussion
Unfortunately we were quite restricted in performing a lot of tests due to low GPU capacity. 
Style transfer generation was slow for the GPU we were able to use. We could have searched for faster methods for style transfer. 

The low GPU performance limited us to using only three different styles. Using different amounts of (abstract and non abstract) styles could have showed us more of a pattern.
Also, we were limited in the amount of test runs. With more runs, we could have done t-tests to see if the improvements are significant.

There is also a potential in improving the results by using different types of augmentation, we have used basic augmentations such as random rotations, horizontal and vertical shifts, zooming, horizontal flips, shear and changes in brightness.

Also, we could have lowered the amount of iterations for generating the style transfer images to see the effect on accuracy scores. Some images, especially the images generated with style 2 became somewhat unrecognizable for a human. This style also performed the worst in both networks.


# 7. Conclusion
Generally, the use of basic augmentation and augmentation based on style transfer both improve the test accuracies for our dataset. However, the style chosen seems to be important for style transfer; style 2 performs even worse than training on only the original dataset. Moreover, using traditional data augmentations is more computationally efficient, while giving similar results to style transfer.

Basic augmentations seem to improve the results, but the strongest improvements are achieved by using both style transfer and basic augmentations. The effect of these improvements seem to be bigger for the VGG16 network, which is a less deep network compared to VGG19. This suggests that using these augmentations can especially be beneficial for shallower networks.

Still, the significance of the results are unclear as we were unable to do t-tests on the results due to low amount of test runs.

# References
[1] Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2018). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231.

[2] Zheng, X., Chalasani, T., Ghosal, K., Lutz, S., & Smolic, A. (2019). Stada: Style transfer as data augmentation. arXiv preprint arXiv:1909.01056.

[3] Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

[4] L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models from few training examples: an incremental Bayesian approach tested on 101 object categories. IEEE. CVPR 2004, Workshop on Generative-Model Based Vision. 2004
