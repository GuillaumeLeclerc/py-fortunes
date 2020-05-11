FROM continuumio/anaconda3 
RUN conda install -y numba matplotlib ipython
RUN mkdir folder
WORKDIR /folder
