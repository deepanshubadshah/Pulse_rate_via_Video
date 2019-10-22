# Pulse_rate_via_Video

Here Pulse rate has been calculated via phone camera video by analysing RGB value change after applying Euler video magnification technique (A computer vision algorithm which was proposed by MIT CS&amp;AI Lab).
http://people.csail.mit.edu/mrub/papers/vidmag.pdf

Key observation points:
1. There was very small change in the red and green color component values and hence I have used blue color component variation for the pulse rate measurement.
2. For a single pulse there is a pair of maximas in the blue component variation.
3. Accuracy has been increased after smoothing.

# Results obtained:

   Net % error in the pulse rate calculation = 3.308%

![Screenshot from 2019-10-22 06-16-25](https://user-images.githubusercontent.com/23376016/67253121-ba088080-f493-11e9-9291-a1f1f2f7b675.png)

# RGB variation with time after magnification and smoothing.
x: time in seconds.
y: Obtained magnified RGB variation (Not RGB values**)

    For 5th result: Pulse rate calculated= 69.09, fitbit reading= 71, error= 2.69% 
![Screenshot from 2019-10-22 06-27-47](https://user-images.githubusercontent.com/23376016/67253424-364f9380-f495-11e9-89e7-e5d1027a1871.png)


    For 4th result: Pulse rate calculated= 80.23, fitbit reading= 80, error= 0.28% 
![Screenshot from 2019-10-22 06-32-44](https://user-images.githubusercontent.com/23376016/67253688-b0cce300-f496-11e9-9b58-f1b6ee53d0a9.png)

    For 7th result: Pulse rate calculated= 87.62, fitbit reading= 87, error= 0.71% 
![Screenshot from 2019-10-22 06-48-18](https://user-images.githubusercontent.com/23376016/67253965-1cfc1680-f498-11e9-8703-f646aacdf6fa.png)

    For 9th result: Pulse rate calculated= 71.48, fitbit reading= 69, error= 3.59% 
![Screenshot from 2019-10-22 06-52-32](https://user-images.githubusercontent.com/23376016/67254072-a9a6d480-f498-11e9-8b1d-cedae620856e.png)
 
