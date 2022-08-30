# **Finding Lane Lines on the Road**

The goals / steps of this project are the following:

Make a pipeline that finds lane lines on the road
Reflect on your work in a written report

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I used the `cv2.cvtColor` method to convert the image to grayscale. I then used the `cv2.GaussianBlur` method to smooth the image to reduce noise. I passed the blurred image to `cv2.Canny` with thresholds of 50 and 150 for low and high, respectively. I chose these numbers via trial and error and these seemed to produce good enough results. After finding the edges, I narrowed down the scope of focus using a mask to limit to the region we expect the lane to fall into. Using the `cv2.HoughLinesP` and `cv2.addWeighted` methods, I was able to draw our detected lines onto our image.

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function by first using the slope of each line segment to determine which side of the lane it represented. A positive slope indicates a line segment on the left side, while a negative slope indicates a line segment on the right side. When selecting line segments, I only accepted left lane line segments with a slope greater than 0.3 (or less than -0.3 for the right lane) to avoid outliers that may impact our final averaged lane line. The selected line segment were then averaged to create what should be the most representative lane line segment. I then extended the line segment through the ordered pair to the bottom and (slightly over) mid-point of the graph.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is the pre-determined min and max parameters for the hough transformation function. I imagine differences in lane markings, such as the distance between dashed lanes, would impact the pipeline's ability to accurately detect lanes. Additionally, the pipeline has some shortcomings when it comes to curves. If the lane turns at an angle great enough to move the lane "end" from the center of the graph, the detected lane lines will not align with reality. It's also possible the lane will move outside the "region_of_interest" we have defined.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to include some logic (not totally sure how!) to understand the curvature of a lane and follow along with it. Another potential improvement could be to use knowledge gained from a prior frame to better understand what we might see in the current and future frames.
