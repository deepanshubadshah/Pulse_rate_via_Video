# Pulse_rate_via_Video

Here Pulse rate has been calculated via phone camera video by analysing RGB value change after applying Euler video magnification technique (A computer vision algorithm which was proposed by MIT CS&amp;AI Lab).
http://people.csail.mit.edu/mrub/papers/vidmag.pdf

Key observation points:
1. There was very small change in the red and green color component values and hence I have used blue color component variation for the pulse rate measurement.
2. For a single pulse there is a pair of maximas in the blue component variation.
3. Accuracy has been increased after smoothing.
