# nuscenes2bag

convert nuscenes data to rosbag for visualization

## Requirements

## Create a virtual environment

```bash
conda create -n nuscenes2bag python=3.8 -y
conda activate nuscenes2bag
cd nuscenes2bag
pip install -r requirements.txt
```

## Usage

```bash
ln -s xxx/nuscenes ./data


python3 nuscenes2bag.py --scene test --version v1.0-trainval --dataroot /home/wind/Projects/ppt_roscenes_ws/nuscenes

python3 nuscenes2bag.py --scene xxx --dataroot ./data/nuscenes --version v1.0-mini --outdir ./data/nuscenes/v1.0-mini

- scene : scene name (ex. scene-0001)
- dataroot: path to the nuScenes dataset
- version: dataset version (v1.0-mini, v1.0-trainval, v1.0-test)

```

## Raw Repo README

> _Convert [nuScenes](https://www.nuscenes.org/) data into ROS [bag](http://wiki.ros.org/rosbag) format_

## Migrated

This project is no longer maintained, it has been superseded by <https://github.com/foxglove/nuscenes2mcap>.

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into ROS bag files for easy viewing in tools such as [Foxglove Studio](https://foxglove.dev/).

## Usage

    # This builds the `nuscenes2bag` Docker image and downloads various nuScenes datasets into `data/`
    $ ./setup.sh

    # Start a Jupyter notebook server on port 8888
    $ ./start.sh

    # Look in the console for the auth token and open a browser to http://localhost:8888/?token=<token>

## License

nuscenes2bag is licensed under [MIT License](https://opensource.org/licenses/MIT).

## Stay in touch

Join our [Slack channel](https://foxglove.dev/join-slack) to ask questions, share feedback, and stay up to date on what our team is working on.
