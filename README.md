# Visual-Odom-Pipeline
Implementing a Visual Odometry pipeline for the course Vision Algorithms for Mobile Robotics by Prof. Dr. Davide Scaramuzza.

## Repository Overview:
- **docs** report images and result GIFs
- **notebooks** notebooks for prototyping
- **results**
	- kitti
	- mala
	- parking
	- outdoor_street
	- roomtour
- **setup** includes conda env
- **src**
	- **bundle_adjuster** bundle adjuster 
	- **camera** camera class
	- **extractor** helper methods for pipeline
	- **loader** loads images for dataset
	- **pipeline** actual pipeline 
	- **state** includes state definitions
	- **visu** visualizer and image logger

## Architecture Overview:
<img src="https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/Pipeline.svg">


## Running the Code: 
0. Navigate to the desired location in the terminal where the repository should be cloned.

1. Clone the repository
```
git clone https://github.com/JonasFrey96/Visual-Odom-Pipeline && cd Visual-Odom-Pipeline
```

2. Installing conda env
```
conda env create -f setup/conda_env.yml
```

3. Activate conda env
```
conda activate vo
```

4. Prepare datasets
The location of the datasets is provided in the datasets.yml. 
Simply modify the paths within the yaml file.
Always provide the path to the main folder of each dataset. 
For using our custom dataset please follow the instruction in the custom dataset instructions section. 

5. Run pipeline
Simply select the dataset with the dataset flag ['parking', 'malaga', 'kitti','roomtour', 'outdoor_street']
```
python src/main.py --dataset=kitti --headless=False
```
If you are using a headless machine (docker container), simply set the headless flag to true.
The results for each dataset will be generated in the results directory. 

These instructions are tested on a linux workstation (Ubuntu 18.04 LTS, Intel i7-7820X, 32GB RAM, Nvidia GTX1080Ti) and laptop (Intel i5, 16GB RAM, Ubuntu 18.04 LTS). 
Please adjust all the commands if you are on a windows and mac machine.

## Results:

Generate videos:
```
sudo apt install imagemagick
cd results/kitti
ffmpeg -r 10 -f image2 -s 1920x1080 -pattern_type glob -i 'out*.png'  -vcodec libx264 -crf 25  -pix_fmt yuv420p output.mp4
ffmpeg -i output.mp4 -r 10 -vf scale=800:400 output.gif
```

**Result Parking Dataset:**

![parking](https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/parking.gif)


**Result Kitti Dataset:**

![kitti](https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/kitti.gif)


**Result Malaga Dataset:**

![malaga](https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/malaga.gif)



**Result Outdoor_Street Dataset:**

![outdoor_street](https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/outdoor_street.gif)


**Result Roomtour Dataset:**

![roomtour](https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/roomtour.gif)



## Custom Datasets:
We uploaded the recorded datasets to PolyBox.
Simply download the dataset to the desired location and put the correct path into the configuration file (datasets.yml) as for the other datasets.

If you are on linux the following commands will download the dataset and untar it.
```
wget -O outdoor_street.tar https://polybox.ethz.ch/index.php/s/GuK0NI5lHXmRqu8/download
wget -O roomtour.tar https://polybox.ethz.ch/index.php/s/5AGLBNuDsn1T0pu/download
tar -xvf outdoor_street.tar
tar -xvf roomtour.tar
```
- Outdoor_Street: https://polybox.ethz.ch/index.php/s/GuK0NI5lHXmRqu8
- Roomtour: https://polybox.ethz.ch/index.php/s/5AGLBNuDsn1T0pu


