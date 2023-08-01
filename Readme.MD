# Visual Python (Concepts, Implimentation and Prototyping)
This repo contains several concepts and implimentations of computer vision and visual slam algorithms for rapid prototyping for reserachers to test concepts.

Some implientations are done with g2o for optimisatiion or Gauss newton non linear solver

For solutions done with Gauss newton code runs very slowly as using the c++/python bind libraries are faster
***
# Cretae virtual environment
```
python3 -m venv venv
```
# G20 installation
On my mac i had to change some things to get to work so eddited g2opy will be attached you can skip the 
git clone
```
git clone https://github.com/uoip/g2opy.git
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install

```

# install py open GL
```
pip install PyOpenGL PyOpenGL_accelerate
```

# Pangolin installation

```
git clone https://github.com/uoip/pangolin.git
cd pangolin
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
```


# Sophus py

```
pip install sophuspy
```

# Open 3d for point clound
```
pip install open3d
```

# Opencv pip install
```
pip install opencv-python
```

# numpy installation 
```
```

# Matplotlib
```
```

# Sci py
```
pip install scipy  
```

