# 1. Edge Detection
 - Convolution process - using filter/kernel
 - filter helps find edges

1 0 -1

2 0 -2
 
1 0 -1

Sobel filter

3 0 -3

10 0 -10
 
3 0 -3

Sharr filter

 - Learning how to detect edges: treat matrix 9 numbers as parameters
 - Backpropagation to learn params
 - Maybe learn edges 45 degrees and others?

# 2. Padding
Each time we apply convolution - image shrinks.

Pixels on the corners are used much less when going over image
with filter.

Padding image - add additional border with 1px - 6x6 becomes
8x8 - convoluted by kernel 3x3 is now 6x6, and not 4x4 like before

result matrix = n-f+1 x n-f+1

now result matrix: n+2p-f+1 x n+2p-f+1

p = padding = 1

uses corners more!

How much padding to add?
- "Valid" - no padding
- "Same" - Pad so that output size is the same as input size

p = (f - 1) / 2

f is usual odd number

# 3. Strided Convolutions

Example: 7x7 * 3x3 filter using stride = 2, makes 2 position step when
moving filter (both vertically and horizontally)

results is: 3x3 matrix

nxn * fxf, padding p , stride s

output = (n+2p-f / s) + 1 x (n+2p-f / s) + 1

(n+2p-f / s) + 1 equals [z]

-> [z] (floor of Z) = rounding if result not int

## Convolution vs cross-correlation:

Convolution in math textbook: Flip matrix both ways (horiz and vertical)


# 4. Convolutions over Volumes
Convolutions on RGB image

Example: 6x6x3 where 3 is number of channels

3D filter needed: 3x3x3 - num of channels must be same for 
image and filter

3x3x3 = 27 numbers -> multiply with 27 corresponding numbers 
on image and sum them up and add to output place

If you want for example to detect edges only in 1 channel,
put 0s in other kernel channels

Multiple filters?
Vertical and horizontal edge detectors - 2 filters 3x3x3
abd 6x6x3 image = we get 4x4x2 because we used 2 filters, would 
be 4x4x1 if we used 1 

NxNxNc * FxFxNc = N-F+1 x N-F+1 x Ncp (num of filters used)

*no stride of padding

# 5. One layer of CNN
Output = image * filter(s)

Output = Relu (Output + bias)

Number of params depends on filters, not images!

Notation: (layer l)
- f[l] -> filter size
- p[l] -> padding
- s[l] -> stride
- Nc[l] -> number of filters
- Input:  N[l-1] x N[l-1] x Nc[l-1] (channels number)
- Output: Nh[l] x Nw[l] x Nc[l]
- Nhw[l] = Round( ( N[l-1] + 2p[l] -f[l] ) / s[l] + 1 )

Each filter: f[l] x f[l] x Nc[l-1]

Activations: a[l] = Nh[l] x Nw[l] x Nc[l]

A[l] = m x Nh[l] x Nw[l] x Nc[l] (mini-batch)

Weights: f[l] x f[l] x Nc[l-1] x Nc[l] (total number of filters) in
layer l

bias: Nc[i] - (1, 1, 1, Nc[l]) 4 dim tensor


# 6. Simple CNN example
Layers:
- Convolution (CONV)
- Pooling (POOL)
- Fully connected (FC)

# 7. Pooling Layers
One variant is Max Pooling - max value from each area

filter and stride size are hyperparameters

detects and persists feature, if number is high some feature exists?

no gradients to learn, just hyperparameters

[ (n + 2p - f ) / s + 1 ] for output dimensions

output has the same number of channels as input!

Other types: Average Pooling (not used as much)

padding rarely used

# 8. CNN examples
Convolutional and Pooling layer can be counted as 1 by convention,
as Pooling layer does not have any parameters

Example: Input -> [ Conv1 -> Pool1 ] -> Conv2 -> Fc3 -> Fc4 -> Output

# 9. Why Convolutions?
Small number of parameters

- Parameter sharing: feature detector that's useful in on part of
the image is probably useful in other parts as well

- Sparsity of connection: In each layer, each output value depends
only on a small number of inputs

# 10. Classic CNNs
1. LeNet - 5, Paper: https://ieeexplore.ieee.org/document/726791
2. AlexNet, Paper: https://dl.acm.org/doi/10.1145/3065386
3. VGG, Paper: https://arxiv.org/abs/1409.1556v6

# 11. Resnets
- Residual block -
follow shortcut instead of main path (skip steps)

Why ResNets work?
Learns Identity Function!!!

# 12. Network in Network
1 x 1 convolution

# 13. Inception Networks
Stacking different volumes, use 1x1 CONV to save on computation
Num of channels is the sum of channel No od all volumes

Side branch - hidden layer with softmax that tries to predict output
Makes sure that even these hidden layers are not bad at predicting
- Regularizing effect

# 14, Transfer Learning
Use someone else's model and weights!

Freezing parameters

Save activations to disk

For larger datasets, freeze a couple of layers only

# 15. Data Augmentation
Mirroring, Cropping, Rotating, Shearing images as example

Color shifting - playing with RGB

Tips for benchmarks/competitions:
- Ensembling (several networks, average outputs)
- Multi-crop and test time (run classifier on multiple version
of the test image, 10-crop for example)

# 16. Object object localization
- 1 object -> classification with localization
- 2 multiple objects -> detection

Another output to output "Bounding Box":
- bx, by - coordinates 
- bh height, bw width
  (percents of the entire image)

Y output vector = [Pc, bx, by, bh, bw, c1, c2, c3...cn]

Pc - 0 or 1 (is there an object on the image?)

ci - label for i-th class

if Pc = 0, we do not care about Bounding Box (bx, by, bh, bw)

loss = sum ( y(i) - yhat (i) ) if y(1) = 1 (Pc)

loss = y(1) - yhat(1) (Pc) if y1 = 0

# 17. Landmark detection
Example: corners of the eye

output = [Pc, l1x, l2x......lnx, lny]

Pc = face ?

l(i)x, l(i)y = x and y for all things you want to detect

(Snapchat filters)

# 18. Object detection
Example: Training set to determine if car or not -> y = 0 | 1

## Sliding Windows detection
- Put rectangle on image, slide windows across every position on the image
- This creates little cropped images - classify them 0 or 1 if car
- Pass larger window (bigger stride)
- Computational cost

# 19. Convolutional Implementation for Sliding Windows
Turning FC layer into convolutional layers

This approach shares computation (test image is larger with padding)

# 20. Building Box Predictions
Sliding window cannot always output the best result, it sometimes cannot
locate object properly due to object size that is different compared to
sliding window

Steps:
- Make grid (3x3 for example and apply localization algorithm)
- For each cell in grid specify y = [Pc, bx, by, bh, bw, c1, c2...cn]
- Again, if Pc=0 we do not care about other values
- Output volume will be 3x3x8 for example, where 8 is the size of vector
[Pc1, bx, by, bh, bw, c1, c2, c3]
- Object's midpoint tells you to which cell it belongs (even if a small
part of it belongs to some other cell)
- bx and by are relative to cell size, always between 0 and 1
- bh and bw can be > 1
- This is YOLO (You Only Live Once Algorithm - paper)

# 21. Intersection over Union
- IOU function computes the size of the intersection and devides it
by side of the union
- Correct if IOU >= 0.5 (object is located)
- it is a measure of the overlap between two bounding boxes

# 22. Nonmax Suppression
Algorithm may detect same object multiple times

You have to make sure it happens only once

Output max IOU, suppress others

Discard all boxes with <= 0.6 probability for example at first

While there are remaining boxes

- Pick largest Pc and output it
- Discard all boxes with IOU <= 0.5 with the box output in previous step

For multiple object, do Nonmax suppression on each one separately

# 23. Anchor Boxes
What to do if grid detects more than 1 object?

More Anchor Boxes for more predictions

Repeat Y output vector multiple times

Example: Y = [Pc bx by bh bw c1 c2 c3 Pc bx by bh bw c1 c2 c3]

2 Anchor Boxes here - Output is 3x3x16


# 24. YOLO Algorithm
Object detection + Nonmax Suppression + Anchor Boxes

# 25. Region Proposals
R-CNN

- Do not slide windows where it does not make sense
- Segmentation algorithm to detect possible objects
- R-CNN is slow
- Fast R-CNN: Propose regions. Use convolution implementation of sliding
windows to classify all the proposed regions
- Faster R-CNN: Use CNN to propose regions

# Bonus: Image Segmentation 
Semantic & Instance segmentation

https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/?utm_source=blog&utm_medium=computer-vision-implementing-mask-r-cnn-image-segmentation


https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/

https://deepsense.ai/region-of-interest-pooling-explained/

# 26. MobileNet
https://medium.com/@godeep48/an-overview-on-mobilenet-an-efficient-mobile-vision-cnn-f301141db94d

# 27. Face Recognition
+ liveness detection (recognize person, not image)
+ Verification vs Recognition
+ Verification 1:1, Check if input is expected output - 1:1
+ Recognition is much harder, 1:K, where database has K person, has to output
id if any of the persons is the one, or "not recognized"

# 28. One Shot Learning
- Learn from only 1 example
- Similarity function -> d(image1, image2) = degree of difference
- if d(image1, image2) < t = same person (t is hyperparameter)

# 29. Siamese Network
- Output layer acts like encoder for the input image - f(x)
- d(x1, x2) = || f(x1) - f(x2) || 2 2 -> is small if it's the same person

# 30. Triplet loss
- Always 3 images (Anchor, Positive, Negative)
- d(A, P) -> to be small, d(A, N) -> to be large
- Norm(f(A), f(P)) / d(A, P) <= Norm(f(A), f(N)) / d(A, N) 
- Norm(f(A), f(P)) - Norm(f(A), f(N)) + alfa <= 0 - alfa
- Alfa is another hyperparameter (called margin) used to avoid triviality
- L(A, P, N) = max([ Norm(f(A), f(P)) - Norm(f(A), f(N)) + alfa <= 0 - alfa ], 0)
- Const function J
- J = sum L(A, P, N) from 1 to m (training set size)
- You need at least 2 images of the same person -> A, P
## Choosing triplets?
- Randomly -> d(A, P) + alfa <= d(A, N) is satisfied easily?
- Better: choose triplets that are hard to train on, where maybe d(A, P) is close to
d(A, N)
- This is needed to push Gradient Descend to separate these two

# 31. Face Verification
- Binary classification
- 2 CNNs outputs -> logistic regression
- target = sigmoid[ sum (wi * (abs(f(x1i) - f(x2i))))+ b ]
- Precompute encodings! for better performace & deployment

# 32. Neural Style Transfer
- C -> content image
- S -> style image
- G = C + S -> generated image
In order to implement, look at extracted features is needed

## What are deep CNNs learning
Pick a unit in layer 1 and 9 images that maximize its activation. Repeat
for other units.

## Const function
* J(G) = cost function for generated image
* J_content(C, G) -> how similar if content of C to that of G
* J_style(S, G) -> how similar is style of S to that of G

J(G) = alfa * J_content(C, G) + beta * J_style(S, G)

alfa, beta - relative weighting between content and style cost

Steps:
1. Init G randomly (100 x 100 x 3)
2. Define J(G)
3. Minimize J(G) with gradient descend

## Content cost function
- Choose layer "l" not too shallow and not too deep
- J_content(C, G) -> how different are activations in layer "l"
- J_content(C, G) = 1/2 Norm([ a[l][c] - a[l][G] ]) (element wise)
- if a[l][C] and a[l][G] are similar, both images have similar content

## Style cost function
Style is defined as correlation between activations across channels

Details and formulas:
https://www.youtube.com/watch?v=QgkLfjfGul8&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=41

# 33. 1D and 3D generalizations

