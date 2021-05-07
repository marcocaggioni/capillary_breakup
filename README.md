# capillary_breakup
Python package for capillary breakup video analysis 

To run the code you can use the docker image at:

```
docker pull mcaggio/jupyterserver
```

to run the image from shell run:

```
docker run -it -p 8888:8888 -v ${PWD}:/home/jovyan/work --user root -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes  mcaggio/jupyterserver start-notebook.sh --NotebookApp.password='sha1:e23cdb41c487:cbf5e2e2339298dcde25787df32d287cfbd506fd'
```

## Steps

* Acquisition -> Video/ Image seq + time stamps & calibration

folder with video or imageseq + xml or json file with metadata

* Processing (goal is to be unsupervised)
  * Identify initial frame - time 0
  * Thresholding -> binary image frame by frame
  * Minimum radius per frame
  * Save csv file
  * Save frame just before breckup
* Analysis
  * Model fitting
  * AI on last frame
