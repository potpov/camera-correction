# k coefficient estimator

this software estimates the first distortion coefficient for an input image.
theory behind this work can be found in this paper: 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1234047

# how to use it

just install requirements
``` pip install -r requirements ```

check the configuration file for parameters and run 
``` python3.7 test.py ```

pay attention to the starting coefficient value and the incremental one,
they can be really different according to your camera.
a bit of human tuning is probably required.

![alt text](https://i.ibb.co/xfVjbVJ/photo-2019-04-29-11-12-40.jpg)