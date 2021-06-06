# Introduction

# Style transfer
Style transfer can be used to compose images in the style of another image, or painting. This specific technique was first introduced by Gatys et al. [2]. It uses a style image (for example of a painting), and an input image, after which it tries to blend these two images together such that the input image is styled like the style image. For this project, three different style images were taken into account, which are shown below.
![all_styles](https://user-images.githubusercontent.com/61514183/120921464-14377300-c6c4-11eb-8544-5c1fc0fe08f1.jpg)
The first and second paintings are more abstract works, where the first painting shows a lot of rounded edges and lines and the second painting contains a lot of rectangles. The third painting is a more classic work of Van Gogh. The goal of this project is to compare the different abstract works to the classical work and see if one sort of painting proves to be more beneficial.

For the style transfer, we use a pretrained VGG19 model on Imagenet, as used by Gatys et al. Intermediate layers of this model are used to obtain the content and style representations. Next, we introduce two losses that are combined and used for the style transfer: content loss and style loss.

## Style loss

## Content loss

## Style transfer results
Example of style transfer results for the classes headphone, water lilly, emu and wildcat are shown below. Here we see that the style transfer is done successfully in most cases. However, for style 2 we see that the original image is less recognizable.
![Screenshot from 2021-06-06 13-27-57](https://user-images.githubusercontent.com/61514183/120922729-105b1f00-c6cb-11eb-94fa-36b9345598b6.png)
![Screenshot from 2021-06-06 13-28-05](https://user-images.githubusercontent.com/61514183/120922730-118c4c00-c6cb-11eb-9830-7f39891ea8e0.png)



# Results

# Discussion

# Conclusion

# References
[1] Zheng, X., Chalasani, T., Ghosal, K., Lutz, S., & Smolic, A. (2019). Stada: Style transfer as data augmentation. arXiv preprint arXiv:1909.01056.
[2] Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.
