import logging
import os
import time
import hydra
from omegaconf import DictConfig
from pathlib import Path

import open3d as o3d
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage

from concept_graphs.utils import load_map, set_seed
from concept_graphs.perception.rgbd_to_pcd import rgbd_to_pcd
from asp_agent.ObjectMapToolbox import ObjectMapToolbox

# A logger for this file
log = logging.getLogger(__name__)


def get_current_object_map(cfg, robot_api, perception_pipeline):
    if cfg.use_map_pickle and cfg.map_path is not None:
        # load pickle if provided
        log.info(f"Loading map from pickle at {cfg.map_path}. Ignoring RobotAPI.")
        object_map = load_map(cfg.map_path + "/map.pkl")
        return object_map

    log.info("Getting current observation from RobotAPI...")

    obs = robot_api.get_rgbd_obs()

    log.info("Mapping current RGBD frame...")
    segments = perception_pipeline(
        obs["rgb"], obs["depth"], obs["intrinsics"], obs["camera_pose"]
    )

    object_map = hydra.utils.instantiate(cfg.mapping)
    object_map.from_perception(**segments)

    if cfg.filter_large_objects:
        object_map.filter_large_objects()

    object_map.filter_min_segments(n_min_segments=cfg.final_min_segments, grace=False)
    object_map.downsample_objects()
    for _ in range(2):
        object_map.denoise_objects()
        object_map.self_merge()
    object_map.downsample_objects()
    object_map.filter_min_points_pcd()

    # also store full scene pcd for collision checking later
    points, rgb = rgbd_to_pcd(**obs)

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(points)
    scene_pcd.colors = o3d.utility.Vector3dVector(rgb.astype(float) / 255.0)

    # voxel downsampling
    scene_pcd = scene_pcd.voxel_down_sample(cfg.scene_pcd_voxel_size)
    object_map.scene_pcd = scene_pcd

    return object_map


@hydra.main(version_base=None, config_path="conf", config_name="viser_agent_server")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # robot api
    robot_api = hydra.utils.instantiate(cfg.robot_api)

    # mapping
    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    perception_pipeline = hydra.utils.instantiate(
        cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor
    )

    vlm_clf = hydra.utils.instantiate(cfg.vlm_clf)
    tagger = hydra.utils.instantiate(cfg.vlm_tag)
    affordance_predictor = hydra.utils.instantiate(cfg.vlm_affordances)
    affordance_pointer = hydra.utils.instantiate(
        cfg.vlm_interactions,
        segmentation_model=segmentation_model,
        vis_dir=str(Path(cfg.output_dir) / "affordance_viz"),
    )

    # agent
    llm_kwargs = {}
    if cfg.agent_llm.model_provider == "google_genai":
        # langchain-google-genai defaults to ADC; pass GEMINI_API_KEY explicitly
        llm_kwargs["google_api_key"] = os.environ["GEMINI_API_KEY"]
    llm = init_chat_model(
        cfg.agent_llm.model,
        model_provider=cfg.agent_llm.model_provider,
        temperature=cfg.agent_llm.temperature,
        **llm_kwargs,
    )
    toolbox = ObjectMapToolbox(
        ft_extractor=ft_extractor,
        vlm_clf=vlm_clf,
        vlm_tagger=tagger,
        affordance_predictor=affordance_predictor,
        affordance_pointer=affordance_pointer,
        center_origin=cfg.center_origin,
        robot_api=robot_api,
    )
    tools = toolbox.make_tools()
    agent = llm.bind_tools(tools)

    system_prompt = (
        "You are a capable robotic agent that uses symbolic reasoning to answer spatial grounding and interaction queries."
        "You can call an object retrieval tool using natural language queries to retrieve objects from the space. Object keys are added to your inventory."
        "When appropriate, you can apply spatial reasoning tools to the objects in your inventory."
        "When requested by the user, you can call the interaction tool to perform actions on the objects."
        "Call the different tools to perform actions and answer the user query."
        "Pay attention to the PRECONDITIONS of each tool call to ensure a valid call."
        "Consider the TIPS of each tool to maximize tool efficiency."
        "When you believe the query has been answered, stop calling tools and provide a final answer."
        "Queries are ALWAYS feasible. If you are unsure, make your best guess."
    )

    # inference loop
    object_map = get_current_object_map(cfg, robot_api, perception_pipeline)
    toolbox.load_object_map(object_map)
    user_input = toolbox.spin_until_agent_query()

    while user_input.lower() != "q":
        log.info(f"Received agent query: {user_input}")

        # show segmentation for a few seconds
        toolbox.viser_server.display_object_segmentation()
        log.info(f"Displaying segmentation...")
        time.sleep(3.0)
        toolbox.viser_server.display_object_rgb()

        # agent loop
        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        while True:
            # Call the model with the current list of messages
            response = agent.invoke(
                messages, config=RunnableConfig(stop_sequences=["\n\n"])
            )

            messages.append(response)

            # Check if the response is a tool call
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    # Find and execute the tool
                    for tool in tools:
                        if tool.name == tool_name:
                            result = tool.invoke(tool_args)
                            messages.append(
                                ToolMessage(
                                    tool_call_id=tool_call["id"], content=str(result)
                                )
                            )
            else:
                # Final answer
                log.info("Final Agent Answer:")
                log.info(response.content)

                log.info(f"Displaying final viser annotations...")
                toolbox.viser_server.clear_reasoning_annotations()
                break

        user_input = toolbox.spin_until_agent_query()
        toolbox.reset()
        object_map = get_current_object_map(cfg, robot_api, perception_pipeline)
        toolbox.load_object_map(object_map)


if __name__ == "__main__":
    main()
