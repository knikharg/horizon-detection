# horizon-detection
Implementation of horizon detection of sample images. The image is rows and columns consisting of pixel values. Horizon detection using Bayes Net and Viterbi is explored. Started by generataing the edge strength map which provides the gradient. Below listed are the two approaches

### Bayes Net:
- In approach 1, on using Bayes Net to detect the ridge line we find maximum likelihood of a row based on the pixel intensity.
- By this assumption, the array of highest intensity pixels indicates the ridge line in the image.
- However, this assumption is not applicable to all cases and this approach also may not work as there can be multiple pixels of high intensity.
- This approach does not take into consideration the prior probability distribution of pixel intensities.

### Viterbi:
- In approach 2, using Viterbi, we solve for maximum a posterior estimate, incorporating emission probability (previous pixel intensity of ridge),
transitional probability(distance from the previous pixel location of ridge) and the previous pixel location of ridge. 
- The initial probability, in this approach remains to be the brightest pixel of the first column, as in the Bayes Net approach.
- In the transitional probability, pij, an added weight is given to those pixels closest to the previous pixel location of ridge,
to increase the probability of moving the ridge along one of those pixels.
