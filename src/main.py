import yaml
import sys
import os
import argparse

from loader import Loader
from pipeline import Pipeline
import logging
import coloredlogs

coloredlogs.install(level='DEBUG')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--yaml-path', type=str, help='Name of dataset yaml file within loader directory.'
                                                    'Used to set paths and configure the different datasets.')
  parser.add_argument('--dataset', type=str, choices=['parking', 'malaga', 'kitti'], help='Name of dataset to use.')
  args = parser.parse_args()

  # setup loader
  with open(os.path.join(sys.path[0], 'loader', args.yaml_path)) as f:
    doc = yaml.load(f, Loader=yaml.FullLoader)
  loader = Loader(args.dataset, doc)
  
  pipeline = Pipeline(loader)
  pipeline.full_run()

