# Visual-Odom-Pipeline
Implementing a Visual Odometry Pipeline for the course Vison Algorithms for Mobile Robotics by Prof. Dr. Davide Scaramuzza ETH Zurich

## Repository Overview:
- **docs** report images and result GIFs
- **notebooks** notebook for prototyping
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
	- **extractor** helper methods for pipleine
	- **loader** loads images for dataset
	- **pipleine** actual pipeline 
	- **state** includes state defintions
	- **visu** visualizer and image logger

## Architecture Overview:
<img src="https://github.com/JonasFrey96/Visual-Odom-Pipeline/blob/master/docs/Pipeline.svg">


## Running the Code: 
0. Navigate to the desired location in the terminal where the repository should be clonde.

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
For useing our costum dataset please follow the instruction in the costume dataset instructions section. 

5. Run pipeline
Simply select the dataset with the dataset flag ['parking', 'malaga', 'kitti','roomtour', 'outdoor_street']
```
python src/main.py --dataset=kitti --headless=False
```
If you are useing a headless machine (docker container), simply set the headless flag to true.
The results for each dataset will be generated in the results directory. 

These insructions are tested on a linux workstation (Ubuntu 18.04 LTS, Intel i7-7820X, 32GB RAM, Nvidia GTX1080Ti) and laptop. 
