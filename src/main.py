import yaml
from loader import Loader
from pipeline import Pipeline
import logging
import coloredlogs

coloredlogs.install(level='DEBUG')

if __name__ == "__main__":
  # setup loader
  with open('/home/jonfrey/Visual-Odom-Pipeline/src/loader/datasets.yml') as f:
    doc = yaml.load(f, Loader=yaml.FullLoader)
  loader = Loader('parking', doc)
  
  pipeline = Pipeline(loader)
  pipeline.full_run()

