import open3d as o3d
from typing import List

from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser

from concept_graphs.viz.utils import similarities_to_rgb
from .const import SKILL_COLOR


class ViserServer:
    def __init__(self, point_shape: str = "circle"):
        self.server = viser.ViserServer()
        self.object_map = None
        self.segmentation_colors = None
        self.point_shape = point_shape
        self.point_cloud_names = []
        self.point_cloud_handles = []
        self.reasoning_annotation_names = []
        self.agent_query = None
        self.clip_query = None

        # gui
        with self.server.gui.add_folder("Agent"):
            self.agent_gui_text = self.server.gui.add_text(
                "Query",
                initial_value="",
            )
            self.agent_gui_button = self.server.gui.add_button(
                "Run Agent",
                icon=viser.Icon.MOUSE,
            )
            self.agent_gui_button.on_click(self.on_query_submit)

        with self.server.gui.add_folder("CLIP Query"):
            self.clip_gui_text = self.server.gui.add_text(
                "Query",
                initial_value="",
            )
            self.clip_gui_button = self.server.gui.add_button(
                "CLIP Similarities",
                icon=viser.Icon.MOUSE,
            )
            self.clip_gui_button.on_click(self.on_clip_query_submit)

        with self.server.gui.add_folder("Point Cloud"):
            self.pcd_rgb_gui_button = self.server.gui.add_button(
                "RGB",
                icon=viser.Icon.MOUSE,
            )
            self.pcd_rgb_gui_button.on_click(self.on_rgb_button_click)

            self.pcd_segmentation_gui_button = self.server.gui.add_button(
                "Segmentation",
                icon=viser.Icon.MOUSE,
            )
            self.pcd_segmentation_gui_button.on_click(self.on_segmentation_button_click)
            self.pcd_size_gui_slider = self.server.gui.add_slider(
                "Point size",
                min=0.001,
                max=0.020,
                step=0.001,
                initial_value=0.003,
                disabled=False,
            )
            self.pcd_size_gui_slider.on_update(self.on_point_size_change)

    # object map update
    def update_object_map(self, object_map):
        self.object_map = object_map
        rng = np.random.default_rng(42)
        self.segmentation_colors = rng.random((len(object_map), 3))

    # gui callbacks and inputs
    def on_query_submit(self, data):
        self.agent_query = self.agent_gui_text.value

    def on_clip_query_submit(self, data):
        self.clip_query = self.clip_gui_text.value

    def on_rgb_button_click(self, data):
        self.display_object_rgb()

    def on_segmentation_button_click(self, data):
        self.display_object_segmentation()

    def on_point_size_change(self, data):
        point_size = self.pcd_size_gui_slider.value

        for handle in self.point_cloud_handles:
            handle.point_size = point_size

    # scene manipulation
    def reset(self):
        self.clear_reasoning_annotations()
        self.server.scene.reset()
        self.display_object_rgb()

    def clear_main_objects(self):
        for name in self.point_cloud_names:
            self.server.scene.remove_by_name(name)
        self.point_cloud_names = []
        self.point_cloud_handles = []

    def clear_reasoning_annotations(self):
        for name in self.reasoning_annotation_names:
            self.server.scene.remove_by_name(name)
        self.reasoning_annotation_names = []

    def display_object_point_clouds(self, colors: np.ndarray = None):
        self.clear_main_objects()

        for i, obj in enumerate(self.object_map):
            name = f"object_{i}"
            # downsample point clouds to make life easier for viser
            pcd = obj.pcd.voxel_down_sample(voxel_size=0.005)
            pcd_points = np.asarray(pcd.points)
            if colors is not None:
                pcd_colors = np.tile(colors[i], (pcd_points.shape[0], 1))
            else:
                # original rgb color
                pcd_colors = np.asarray(pcd.colors)
            handle = self.server.add_point_cloud(
                name,
                pcd_points,
                pcd_colors,
                point_size=self.pcd_size_gui_slider.value,
                point_shape=self.point_shape,
            )
            self.point_cloud_names.append(name)
            self.point_cloud_handles.append(handle)

    def display_object_rgb(self):
        self.display_object_point_clouds()

    def display_object_segmentation(self):
        self.display_object_point_clouds(colors=self.segmentation_colors)

    def display_object_similarity(self, similarity_scores: np.ndarray):
        self.display_object_point_clouds(
            colors=similarities_to_rgb(similarity_scores, cmap_name="viridis")
        )

    def add_oriented_bbox(self, object_index: int, label: str):
        # bbox around object + label
        bbox = self.object_map[object_index].pcd.get_oriented_bounding_box()
        center = bbox.center
        extent = bbox.extent
        R = np.array(bbox.R, copy=True)

        quat = Rsc.from_matrix(R).as_quat()  # x, y, z, w
        wxyz = (quat[3], quat[0], quat[1], quat[2])  # w, x, y, z

        name = f"bbox_{object_index}"
        self.server.scene.add_box(
            name=name,
            color=(0, 0, 0),
            dimensions=extent,
            position=center,
            wxyz=wxyz,
            visible=True,
            wireframe=True,
        )
        self.reasoning_annotation_names.append(name)

        if label != "":
            label_name = f"bbox_label_{object_index}"
            label_pos = center.copy()

            self.server.scene.add_label(
                name=label_name,
                text=label,
                position=label_pos,
                visible=True,
            )
            self.reasoning_annotation_names.append(label_name)

    def add_square_markers(
        self,
        object_indices: List[int],
        color: tuple = (64, 64, 64),
        is_temp=True,
        name_suffix: str = "",
    ):
        # square marker under objects
        for obj_idx in object_indices:
            obj = self.object_map[obj_idx]
            pcd = np.asarray(obj.pcd.points)
            centroid = obj.centroid

            z = pcd[:, 2].min() - 0.01
            x_len = pcd[:, 0].max() - pcd[:, 0].min()
            y_len = pcd[:, 1].max() - pcd[:, 1].min()
            side_len = max(x_len, y_len) * 0.95
            square_points = np.array(
                [
                    [-side_len / 2, -side_len / 2, 0],
                    [side_len / 2, -side_len / 2, 0],
                    [side_len / 2, side_len / 2, 0],
                    [-side_len / 2, side_len / 2, 0],
                    [-side_len / 2, -side_len / 2, 0],
                ]
            )
            square_points += centroid
            square_points[:, 2] = z

            segments = np.stack([square_points[:-1], square_points[1:]], axis=1)

            name = f"square_{obj_idx}_{name_suffix}"
            self.server.scene.add_line_segments(
                name=name,
                points=segments,
                colors=color,
                line_width=4.0,
                visible=True,
            )

            if is_temp:
                self.reasoning_annotation_names.append(name)

    def add_interact_markers(
        self, object_index: int, aff_pcd: np.ndarray, aff_type: str
    ):
        # to open3d and downsample
        aff_pcd_o3d = o3d.geometry.PointCloud()
        aff_pcd_o3d.points = o3d.utility.Vector3dVector(aff_pcd)
        aff_pcd_o3d = aff_pcd_o3d.voxel_down_sample(voxel_size=0.005)
        aff_pcd = np.asarray(aff_pcd_o3d.points)

        # affordance pcd
        color = (np.array(SKILL_COLOR[aff_type]) * 255).astype(np.uint8).tolist()
        colors = np.tile(color, (aff_pcd.shape[0], 1))

        aff_handle = self.server.add_point_cloud(
            name=f"aff_{object_index}",
            points=aff_pcd,
            colors=colors,
            point_size=self.pcd_size_gui_slider.value * 2.0,
            point_shape=self.point_shape,
        )
        self.point_cloud_handles.append(aff_handle)

        # aff_type label
        pos = self.object_map[object_index].centroid
        pos[2] += 0.10
        self.server.scene.add_label(
            name=f"aff_label_{object_index}",
            text=aff_type,
            position=pos,
            visible=True,
        )

        # change square marker color
        self.server.scene.remove_by_name(f"square_{object_index}")
        self.add_square_markers(
            [object_index], color=color, is_temp=False, name_suffix="interact"
        )

    def add_segment_between(
        self,
        object_idx_1: int,
        object_idx_2: int,
        color: tuple = (0, 0, 255),
        label: str = "",
    ):
        # line between centroids of the two objects
        obj1 = self.object_map[object_idx_1]
        obj2 = self.object_map[object_idx_2]
        centroid1 = obj1.centroid
        centroid2 = obj2.centroid

        points = np.stack([centroid1, centroid2], axis=0).reshape((1, 2, 3))

        name_line = f"line_{object_idx_1}_{object_idx_2}_{label}"
        self.server.scene.add_line_segments(
            name=name_line,
            points=points,
            colors=color,
            line_width=4.0,
            visible=True,
        )

        # add distance label
        midpoint = (centroid1 + centroid2) / 2
        name_label = f"label_{object_idx_1}_{object_idx_2}_{label}"
        self.server.scene.add_label(
            name=name_label,
            text=label,
            position=midpoint,
            visible=True,
        )

        # spheres at centroids
        name_sphere_1 = f"sphere_{object_idx_1}"
        self.server.scene.add_icosphere(
            name=name_sphere_1,
            color=color,
            radius=0.005,
            position=centroid1,
        )
        name_sphere_2 = f"sphere_{object_idx_2}"
        self.server.scene.add_icosphere(
            name=name_sphere_2,
            color=color,
            radius=0.005,
            position=centroid2,
        )

        self.reasoning_annotation_names.extend(
            [name_line, name_label, name_sphere_1, name_sphere_2]
        )
