# [IROS 2026] Agentic Scene Policies (ASP)
[**Project Page**](https://montrealrobotics.ca/agentic-scene-policies.github.io/) |
[**ArXiv**](https://arxiv.org/abs/2509.19571)

[Sacha Morin](https://sachamorin.github.io/), [Kumaraditya Gupta](https://www.kumaradityag.com/), [Mahtab Sandhu](https://scholar.google.com/citations?user=Gdv8B50AAAAJ&hl=en), [Charlie Gauthier](https://velythyl.github.io/), [Francesco Argenziano](https://www.linkedin.com/in/fra-arg/), [Kirsty Ellis](https://mila.quebec/en/directory/kirsty-ellis), [Liam Paull](http://liampaull.ca)

Perception and agent code for our IROS 2026 paper [Agentic Scene Policies](https://arxiv.org/abs/2509.19571). 

<p align="left">
  <img src="https://sachamorin.github.io/images/asp.gif" alt="ASP Pipeline" width="60%">
</p>

# Install
Clone repository and submodules.
```bash
git clone --recurse-submodules git@github.com:montrealrobotics/agentic-scene-policies.git
```
If you already cloned without `--recurse-submodules`, run `git submodule update --init --recursive` from the repository root.

Install dependencies. Preferably in a virtual environment.
```bash
pip install --upgrade pip
```
```bash
pip install -e agentic-scene-policies
```
```bash
pip install -e agentic-scene-policies/rgbd_dataset
```

## Paths 
Update `conf/paths/paths.yaml`.

```yaml
# @package _global_
cache_dir: ??? # Where to save model checkpoints and other assets
data_dir: ??? # Where to look for datasets by default
output_dir: ??? # Where to save outputs
```
Alternatively, you can create your own `conf/paths/my_paths.yaml` and append `paths=my_paths` to all commands.

## Model Checkpoints
Create ```cache_dir``` and download the following to it, as defined in your config.
 * [mobile_sam.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt)
 * [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
 * [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)
 * [scannet200_classes.txt](https://raw.githubusercontent.com/concept-graphs/concept-graphs/66175d63f466d264edce9f1fb6987c5ba1dcac0e/conceptgraph/scannet200_classes.txt)

**Note 0**: The CLIP checkpoints are downloaded automatically to `cache_dir` on the first run. `CLIP_H14`, the default, is about 4 GB.

**Note 1**: The ASP experiments only used Mobile SAM. Yolo was not used for the ASP experiments but can be useful to process larger scenes more quickly using `algo=CGDetector`.

**Note 2**: If you wish to use `YoloSAM2` for segmentation, also install SAM2 from their repository found [here](https://github.com/facebookresearch/sam2).

## Data

### ASP Lab Data
Download our [lab data](https://drive.google.com/file/d/1LHmorGDR3-83w5vrHlZAupo3vkTAfFo7/view?usp=sharing) and extract it to `data_dir`. Most scenes contain a single frame.

### Replica
You can download the Replica dataset to `data_dir` using the 
[download script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) from Nice-SLAM.

## API Keys
By default, the mapping pipeline (captioning, tagging, affordances, interactions) and the agent LLM all use Gemini and require a `GEMINI_API_KEY` in your environment. Some [options](#building-the-object-map) may require an `OPENAI_API_KEY`.

# Usage
## Map & Query
A single command to run the mapping pipeline on one of the ASP scenes and run some agent queries using a web viewer.
```bash
python3 viser_agent_server.py algo=ASP dataset=ASP dataset.scene=deskbell_drawer
```
Then open `http://localhost:8080/` in a browser and try out some queries (e.g., `ring the desk bell`). The viewer displays various visual markers corresponding to agent tool calls.

Every time you input an agent query, the script (i) runs the perception pipeline on the current RGB-D frame to extract an `ObjectMap`, and (ii) runs the ASP agent to execute the user instruction. This setup emulates online inference with a real robot camera. 

The code currently relies on an `OfflineRobotAPI` which (i) loads RGB-D frames from an offline dataset and (ii) logs skill tool calls without otherwise acting.

## Decoupled Map & Query (ASP lab data)
To save inference time, you can first pickle an `ObjectMap` then run the agent on the pickle.

```bash
python3 main.py algo=ASP dataset=ASP dataset.scene=mugs_toys
```

```bash
python3 viser_agent_server.py use_map_pickle=true
```
Then open `http://localhost:8080/` in a browser for viser.

`viser_agent_server.py` will default to `output_dir/latest_map` which is created by `main.py`. You can alternatively specify `map_path=$MAP_PATH`. 

## Decoupled Map & Query (Replica)
You can also map multiple RGB-D frames from larger scenes by configuring the right dataset. For example
```bash
python3 main.py algo=CGDetector dataset=Replica_low dataset.scene=room0 sim_thresh=0.89 
```
```bash
python3 viser_agent_server.py use_map_pickle=true
```
Then open `http://localhost:8080/` in a browser for viser.


# Advanced Mapping Guide
## Building the Object Map

Mapping configs are defined with [Hydra](https://hydra.cc/docs/intro/). 

We define specific experiments with the `algo` and `dataset` keys. To run the detector variant of Concept-Graphs on a low-resolution
version of the Replica dataset, try

```bash
python3 main.py algo=CGDetector dataset=Replica_low sim_thresh=0.89
```
The map and other assets will be saved to `output_dir`. `main.py` will also create
a symlink to the latest output in `output_dir/latest_map`.

For the slower, SAM-only variant, try `algo=CG` with `sim_thresh=0.85`. `algo=ASP` includes the tuned parameters for the ASP experiments.

## Mapping Parameters

The following arguments can be added to the main command.

* `sim_thresh=0.9`: Similarity threshold between 0 and 1 to merge objects. A high threshold generally gives better object definition, but also a higher occurence of oversegmented or duplicated objects.
* `overlap_eps=0.025`: The radius used for computing the geometric similarity (point cloud overlap). Generally, using the same value as `voxel_size` is a good default.
* `voxel_size=0.025`: The voxel size used for downsampling the object point clouds.
* `denoising_eps=0.1`: The epsilon parameter of the DBSCAN denoising callback. Setting this to 3 or 4 times the `voxel_size` is a good default.
* `max_points_pcd=8000`: The maximum number of points used per object for computing the geometric similarity. A lower number will make the program faster, but the geometric similarity less accurate, especially for large objects.
* `final_min_segments=5`: The minimum number of times an object should be detected to be kept in the map. Helpful to get rid of "noisy" segments that only appear once or twice in the sequence.
* `caption=true`: Ask a VLM to caption the objects. By default, we use Gemini models and you need a `GEMINI_API_KEY` environment variable for this to work. To use OpenAI instead, override with `vlm_caption=OpenAICaptioner` (requires `OPENAI_API_KEY`).
* `tag=true`: Ask a VLM to tag the objects. By default, we use Gemini models and you need a `GEMINI_API_KEY` environment variable for this to work. To use OpenAI instead, override with `vlm_tag=OpenAITagger` (requires `OPENAI_API_KEY`).
* `affordances=true`: Ask a VLM to predict affordance tags on each object. By default, we use Gemini models and you need a `GEMINI_API_KEY` environment variable for this to work.
* `interactions=true`: Get affordance instances and 3D interaction points on each object with the help of a VLM and a segmentation model. By default, we use Gemini models and you need a `GEMINI_API_KEY` environment variable for this to work.
* `device=cuda`: Torch device for the perception models and mapping algo.
* `debug=true`: Save additional visualizations of the segmentations.
* `save_map=true`: Save the map. See the [Output](#output) section.
* `seed=123`: Seed.

Additionally, you can also use all the dataset arguments detailed in the [rgbd_dataset README](https://github.com/sachaMorin/rgbd_dataset).

The above list only includes the most common arguments. If you understand [Hydra](https://hydra.cc/docs/intro/), there are a lot more options that you can configure from the CLI. Add `--cfg job` to the main command to visualize the full config.


## Object Map Output
The output consists of the following files and directories:
* `config.yaml`: The full config of this map.
* `map.pkl`: A pickle object containing the full map.
* `stats.json`: Mapping runtime statistics (fps, mapping time, number of objects and frames).
* `clip_features.npy`: The object CLIP features as an `(n_objects, n_dims)` array.
* `point_cloud.pcd`: The complete point cloud.
* `segments_anno.json`: The object annotations following the Scannet++ format. This includes the point indices in the main point cloud, the caption and the tag of every object. Look at [this method](https://github.com/sachaMorin/concept-nodes/blob/791677da1f3de3e007fff0b4bb8f1478d2fe0c61/visualizer.py#L172) for an example of how to load object point clouds.
* `segments`: The top RGB and mask crops for every object. By default, the code keeps the crops with the highest mask resolution.
* `object_viz`: Visualization of the RGB crops, the caption and the tag for every object.
* `debug`: If `debug=true`, additional visualizations of the segmentations.


# Robot Code
We do not plan to release the robot code at this time. A robot integration would require implementing and configuring a custom `RobotAPI`.

# Acknowledgements
The ASP object map code started as a reimplementation of some ConceptGraphs functionalities (without edges). Check out [the ConceptGraphs website](https://concept-graphs.github.io/) and [the original code](https://github.com/concept-graphs/concept-graphs).


# Reference
If you find this code useful, consider citing
```bibtex
@misc{morin2025asp,
      title={Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action}, 
      author={Sacha Morin and Kumaraditya Gupta and Mahtab Sandhu and Charlie Gauthier and Francesco Argenziano and Kirsty Ellis and Liam Paull},
      year={2025},
      eprint={2509.19571},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19571}, 
}
```
