import logging
import time
from typing import List
import numpy as np
import torch

import open3d as o3d
from langchain_core.tools import tool

from .base_models import State, ToolOutput
from .ViserServer import ViserServer


log = logging.getLogger(__name__)


class ObjectMapToolbox:
    def __init__(
        self,
        ft_extractor,
        vlm_clf,
        vlm_tagger,
        affordance_predictor,
        affordance_pointer,
        center_origin,
        robot_api=None,
    ):
        self.object_map = None
        self.ft_extractor = ft_extractor
        self.vlm_clf = vlm_clf
        self.vlm_tagger = vlm_tagger
        self.affordance_predictor = affordance_predictor
        self.affordance_pointer = affordance_pointer
        self.robot_api = robot_api

        self.device = ft_extractor.device

        # State
        self.object_buffer = dict()
        self.held_object = None

        # Use average of the scene point cloud as the origin for left/right
        # reasoning. Computed in load_object_map, once a map is available.
        self.center_origin = center_origin
        self.origin = np.array([0.0, 0.0])

        # Viser
        self.viser_server = ViserServer()

    @property
    def inventory(self) -> List[str]:
        return list(self.object_buffer.keys())

    @property
    def state(self) -> State:
        return State(held_object=self.held_object, inventory=self.inventory)

    def load_object_map(self, object_map):
        self.object_map = object_map
        self.object_map.to(self.device)

        if self.center_origin and len(object_map):
            all_points = np.vstack([np.asarray(obj.pcd.points) for obj in object_map])
            self.origin = all_points.mean(axis=0)[:2]  # Only xy

        self.viser_server.update_object_map(object_map)
        self.reset()

    def reset(self):
        self.object_buffer = dict()
        self.held_object = None
        self.viser_server.reset()

    def _clip_query(self, query: str) -> torch.Tensor:
        query_ft = self.ft_extractor.encode_text([query])
        query_ft = query_ft.to(self.device)
        sim_objects = self.object_map.similarity.semantic_similarity(
            query_ft, self.object_map.semantic_tensor
        )[0]

        self.viser_server.display_object_similarity(sim_objects.cpu().numpy())

        return sim_objects

    def make_object_retrieval_tool(self, top_k=3):
        @tool
        def object_retrieval(query: str) -> ToolOutput:
            """Retrieve the relevant objects in the scene given the query and adds them to the inventory.

            TIPS:
            - Query for only one object type at a time.
            - Be specific when you can. For example, if you want to find a "red ball", query for "red ball" instead of just "ball".
            - Do NOT query the workspace with object parts (e.g., "handle" when you want to grab a "mug").
            - Do NOT query for objects that are already in your inventory.
            - Do NOT add spatial modifiers to your query (e.g., "left", "right", "nearest", "farthest", "next to"). Use the spatial reasoning tools for that.

            Args:
                query (str): The query string to search for.

            Returns:
                ToolOutput: The output containing the updated state.
            """
            sim_objects = self._clip_query(query)

            self.viser_server.display_object_similarity(sim_objects.cpu().numpy())

            indices = torch.argsort(sim_objects, descending=True).cpu().tolist()[:top_k]

            # Verify top k CLIP objects with VLM
            segments = [self.object_map[i].segments.get_sorted()[0] for i in indices]
            rgbs = [s.rgb for s in segments]
            relevant = self.vlm_clf(rgbs, query)

            for i, is_relevant in enumerate(relevant):
                if is_relevant:
                    object_key = f"{query}_{i}"
                    self.object_buffer[object_key] = indices[i]

            # Return ToolOutput
            if np.any(relevant):
                out = ToolOutput(success=True, output=self.state)
            else:
                out = ToolOutput(
                    success=False,
                    output=self.state,
                    feedback_msg="No relevant objects found for the query.",
                )

            self.viser_server.add_square_markers(
                [indices[i] for i, rel in enumerate(relevant) if rel]
            )

            self.viser_server.display_object_rgb()

            return out

        return object_retrieval

    def make_distance_between_tool(self):
        @tool
        def distance_between(object_key_1: str, object_key_2: str) -> ToolOutput:
            """Compute the Euclidean distance between two objects in the inventory.

            PRECONDITIONS:
            - Both object_key_1 and object_key_2 must be present in the inventory.

            TIPS:
            - Use this tool to compare distances between two objects.
            - Use this tool to assess 'next to' relationships. Next to objects usually have a distance of less than 1.0 meters.

            Args:
                object_key_1 (str): The key of the first object in the inventory.
                object_key_2 (str): The key of the second object in the inventory.

            Returns:
                ToolOutput: The output containing the distance as a float.
            """
            if (
                object_key_1 not in self.object_buffer
                or object_key_2 not in self.object_buffer
            ):
                return ToolOutput(
                    success=False,
                    feedback_msg="ERROR! One or both object keys are not in the inventory. Please provide valid object keys.",
                )

            obj1 = self.object_map[self.object_buffer[object_key_1]]
            obj2 = self.object_map[self.object_buffer[object_key_2]]

            centroid1 = obj1.centroid
            centroid2 = obj2.centroid

            distance = np.linalg.norm(centroid1 - centroid2)

            self.viser_server.add_segment_between(
                self.object_buffer[object_key_1],
                self.object_buffer[object_key_2],
                label=f"{distance:.2f} m",
            )

            return ToolOutput(success=True, output=distance)

        return distance_between

    def _is_left_of(self, obj_1, obj_2):
        vec_1 = obj_1.centroid[:2] - self.origin
        vec_2 = obj_2.centroid[:2] - self.origin

        # 2D cross product: scalar result tells you the relative orientation
        cross = vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]

        return cross < 0  # object_1 is to the left of object_2

    def make_is_left_of_tool(self):
        @tool
        def is_left_of(object_key_1: str, object_key_2: str) -> ToolOutput:
            """Determine if the first object is to the left of the second object from the origin's perspective.

            PRECONDITIONS:
            - Both object_key_1 and object_key_2 must be present in the inventory.

            TIPS:
            - Use this tool to assess 'left of' relationships.

            Args:
                object_key_1 (str): The key of the first object in the inventory.
                object_key_2 (str): The key of the second object in the inventory.

            Returns:
                ToolOutput: The output containing the result of the operation as a boolean.
            """
            if (
                object_key_1 not in self.object_buffer
                or object_key_2 not in self.object_buffer
            ):
                return ToolOutput(
                    success=False,
                    output="ERROR! One or both object keys are not in the inventory. Please provide valid object keys.",
                )

            obj1 = self.object_map[self.object_buffer[object_key_1]]
            obj2 = self.object_map[self.object_buffer[object_key_2]]

            self.viser_server.add_segment_between(
                self.object_buffer[object_key_1],
                self.object_buffer[object_key_2],
                label=f"LEFT OF",
            )

            is_left_of = self._is_left_of(obj1, obj2)
            feedback_msg = f"Object '{object_key_1}' is {'LEFT' if is_left_of else 'RIGHT'} of object '{object_key_2}'."

            return ToolOutput(
                success=True, output=is_left_of, feedback_msg=feedback_msg
            )

        return is_left_of

    def make_is_right_of_tool(self):
        @tool
        def is_right_of(object_key_1: str, object_key_2: str) -> ToolOutput:
            """Determine if the first object is to the right of the second object from the origin's perspective.

            PRECONDITIONS:
            - Both object_key_1 and object_key_2 must be present in the inventory.

            TIPS:
            - Use this tool to assess 'right of' relationships.

            Args:
                object_key_1 (str): The key of the first object in the inventory.
                object_key_2 (str): The key of the second object in the inventory.
            Returns:
                ToolOutput: A ToolOutput object containing the result of the operation as a boolean.
            """
            if (
                object_key_1 not in self.object_buffer
                or object_key_2 not in self.object_buffer
            ):
                return ToolOutput(
                    success=False,
                    output="ERROR! One or both object keys are not in the inventory. Please provide valid object keys.",
                )

            obj1 = self.object_map[self.object_buffer[object_key_1]]
            obj2 = self.object_map[self.object_buffer[object_key_2]]

            self.viser_server.add_segment_between(
                self.object_buffer[object_key_1],
                self.object_buffer[object_key_2],
                label=f"RIGHT OF",
            )

            is_right_of = not self._is_left_of(obj1, obj2)
            feedback_msg = f"Object '{object_key_1}' is {'RIGHT' if is_right_of else 'LEFT'} of object '{object_key_2}'."

            return ToolOutput(
                success=True, output=is_right_of, feedback_msg=feedback_msg
            )

        return is_right_of

    def _size_of(self, obj) -> float:
        lx, ly, lz = obj.pcd.get_oriented_bounding_box().extent
        return lx * ly * lz

    def make_size_of_tool(self):
        @tool
        def size_of(object_key: str) -> ToolOutput:
            """Compute the size (volume) of an object in the inventory.

            PRECONDITIONS:
            - object_key must be present in the inventory.

            TIPS:
            - Use this tool to compare sizes between two objects.
            - Use this tool to assess 'bigger than' or 'smaller than' relationships.

            Args:
                object_key (str): The key of the object in the inventory.

            Returns:
                ToolOutput: The output containing the size as a float.
            """
            if object_key not in self.object_buffer:
                return ToolOutput(
                    success=False,
                    feedback_msg="ERROR! The object key is not in the inventory. Please provide a valid object key.",
                )

            obj = self.object_map[self.object_buffer[object_key]]
            size = self._size_of(obj)

            self.viser_server.add_oriented_bbox(
                self.object_buffer[object_key], label=f"SIZE: {size:.4f} m³"
            )

            return ToolOutput(success=True, output=size)

        return size_of

    def make_place_tool(self):
        """Place currently held object into or on top of another object."""

        @tool
        def place(object_key: str) -> ToolOutput:
            """
            Place the object you are currently holding into or on top of the object with object_key.

            PRECONDITIONS:
            - object_key must be present in the inventory.
            - held_object should not be None. You should be holding an object when placing.

            Parameters:
                object_key (str): The key of the object to place the current object in or on. Should be in the inventory.

            Returns:
                CallOutput: An object containing the success status, status message, and the current state of the robot.
            """
            if object_key not in self.object_buffer:
                return ToolOutput(
                    success=False,
                    feedback_msg="ERROR! The object key is not in the inventory. Please provide a valid object key.",
                )
            if self.held_object is None:
                return ToolOutput(
                    success=False,
                    feedback_msg="ERROR! You are not holding any object to place. Please pick up an object first with the interact tool.",
                )

            self.held_object = None

            pcd = np.asarray(self.object_map[self.object_buffer[object_key]].pcd.points)

            success, feedback_msg = self.robot_api.run_skill("PLACE", pcd, pcd)

            self.viser_server.add_interact_markers(
                object_index=self.object_buffer[object_key],
                aff_pcd=pcd,
                aff_type="PLACE",
            )

            return ToolOutput(
                success=success, output=self.state, feedback_msg=feedback_msg
            )

        return place

    def make_interact_tool(self):
        """Interact with an object in the inventory using more advanced skills."""

        @tool
        def interact(object_key: str, action: str) -> ToolOutput:
            """
            Interact with an object in your inventory.

            PRECONDITIONS:
            - object_key must be present in the inventory.
            - held_object should be None. You should not be holding any object when interacting.

            TIPS:
            - This tool allows you to perform actions such as grasping, pulling, pushing or opening an object.

            Parameters:
                object_key (str): The key of the object to interact with. Should be in the inventory.
                action (str): A plain description of what you aim to achieve with this object based on the user query.

            Returns:
                ToolOutput: An object containing the success status, status message, and the current state of the robot.
            """
            if object_key not in self.object_buffer:
                return ToolOutput(
                    success=False,
                    feedback_msg="ERROR! The object key is not in the inventory. Please provide a valid object key.",
                )
            if self.held_object is not None:
                return ToolOutput(
                    success=False,
                    feedback_msg="ERROR! You are already holding an object. Please place it first before interacting with another object.",
                )

            # ObjectMap with only the relevant object
            object_index = self.object_buffer[object_key]
            scene_obj = self.object_map.subset([object_index])

            # Detect affordance
            self.vlm_tagger.caption_map(scene_obj)
            self.affordance_predictor.user_query = f" The user asks you to '{action}'. Think step-by-step about the affordances of the object that are relevant to the user query."
            self.affordance_predictor.annotate_map(scene_obj)
            self.affordance_pointer.annotate_map(scene_obj, skip_aff_types=["GRASP"])

            affordances = scene_obj[0].affordance_instances
            if len(affordances) == 0:
                return ToolOutput(
                    success=False,
                    output=self.state,
                    feedback_msg="ERROR! No affordance was grounded on this object. Try another object key, or rephrase the action.",
                )

            obj_pcd = np.asarray(scene_obj[0].pcd.points)
            first_aff = next(iter(affordances.values()))
            aff_type = first_aff.type
            aff_pcd = first_aff.mask_3d

            success, feedback_msg = self.robot_api.run_skill(aff_type, obj_pcd, aff_pcd)

            self.viser_server.add_interact_markers(
                object_index=object_index, aff_pcd=aff_pcd, aff_type=aff_type
            )

            # Tool Output
            if success and aff_type in ["GRASP", "GRASP PART"]:
                self.held_object = object_key

            if success and aff_type == "PLACE":
                self.held_object = None

            return ToolOutput(
                success=success, output=self.state, feedback_msg=feedback_msg
            )

        return interact

    def make_tools(self):
        # Object retrieval
        object_retrieval_tool = self.make_object_retrieval_tool()

        # Spatial
        distance_between_tool = self.make_distance_between_tool()
        is_left_of_tool = self.make_is_left_of_tool()
        is_right_of_tool = self.make_is_right_of_tool()
        size_of_tool = self.make_size_of_tool()

        # Interact
        place = self.make_place_tool()
        interact = self.make_interact_tool()

        return [
            object_retrieval_tool,
            distance_between_tool,
            is_left_of_tool,
            is_right_of_tool,
            size_of_tool,
            place,
            interact,
        ]

    # server spin
    def spin_until_agent_query(self):
        log.info("Agent ready! Waiting for agent or CLIP query submission...")

        while self.viser_server.agent_query is None:

            if self.viser_server.clip_query is not None:
                clip_query = self.viser_server.clip_query
                self.viser_server.clip_query = None
                print(f"Received CLIP query: {clip_query}")
                self._clip_query(clip_query)

            time.sleep(0.02)

        query = self.viser_server.agent_query
        self.viser_server.agent_query = None
        print(f"Received query: {query}")
        return query
