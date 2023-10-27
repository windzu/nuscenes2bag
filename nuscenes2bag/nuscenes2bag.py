import argparse
import math
import os
import random
from pprint import pprint
from typing import Dict, List, Tuple

import numpy as np
import rosbag
import rospy
import seaborn as sns
import yaml
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped
from matplotlib import pyplot as plt
from nav_msgs.msg import OccupancyGrid, Odometry
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pypcd import numpy_pc2, pypcd
from pyquaternion import Quaternion
from sensor_msgs.msg import (
    CameraInfo,
    CompressedImage,
    Imu,
    NavSatFix,
    PointCloud2,
    PointField,
)
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import ImageMarker, Marker, MarkerArray

from .utils import (
    BitMap,
    derive_latlon,
    find_closest_lidar,
    get_basic_can_msg,
    get_camera,
    get_camera_info,
    get_centerline_markers,
    get_imu_msg,
    get_lidar,
    get_lidar_imagemarkers,
    get_odom_msg,
    get_pose,
    get_remove_imagemarkers,
    get_scene_map,
    get_tfmessage,
    get_time,
    get_transform,
    get_utime,
    make_color,
    write_boxes_imagemarkers,
    write_occupancy_grid,
)


class Nuscenes2Bag:
    def __init__(
        self,
        scene,
        version,
        dataroot,
        lidar_channel="LIDAR_TOP",
        use_map=False,
        use_can=False,
    ):
        self.scene_name = scene
        self.NUSCENES_VERSION = version
        self.dataroot = dataroot

        self.lidar_channel = lidar_channel

        self.use_map = use_map
        self.use_can = use_can

        self.nusc = NuScenes(
            version=self.NUSCENES_VERSION, dataroot=self.dataroot, verbose=True
        )
        # self.nusc_can = NuScenesCanBus(dataroot='data')

        self.nusc.list_scenes()

    def convert(self):
        scene = self.nusc.scene[0]
        self.convert_scene(scene)

    def convert_scene(self, scene):
        scene_name = scene["name"]
        log = self.nusc.get("log", scene["log_token"])
        location = log["location"]

        # TODO(wind):暂时没有map
        if self.use_map:
            print(f'Loading map "{location}"')
            nusc_map = NuScenesMap(dataroot="data", map_name=location)
            print(f'Loading bitmap "{nusc_map.map_name}"')
            bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, "basemap")
            print(f"Loaded {bitmap.image.shape} bitmap")

        cur_sample = self.nusc.get("sample", scene["first_sample_token"])

        if self.use_can:
            can_parsers = [
                [self.nusc_can.get_messages(scene_name, "ms_imu"), 0, get_imu_msg],
                [self.nusc_can.get_messages(scene_name, "pose"), 0, get_odom_msg],
                [
                    self.nusc_can.get_messages(scene_name, "steeranglefeedback"),
                    0,
                    lambda x: get_basic_can_msg("Steering Angle", x),
                ],
                [
                    self.nusc_can.get_messages(scene_name, "vehicle_monitor"),
                    0,
                    lambda x: get_basic_can_msg("Vehicle Monitor", x),
                ],
                [
                    self.nusc_can.get_messages(scene_name, "zoesensors"),
                    0,
                    lambda x: get_basic_can_msg("Zoe Sensors", x),
                ],
                [
                    self.nusc_can.get_messages(scene_name, "zoe_veh_info"),
                    0,
                    lambda x: get_basic_can_msg("Zoe Vehicle Info", x),
                ],
            ]

        bag_name = f"NuScenes-{self.NUSCENES_VERSION}-{scene_name}.bag"
        # bag_path = os.path.join(os.path.abspath(os.curdir), bag_name)
        bag_path = os.path.join(self.dataroot, bag_name)
        print(f"Writing to {bag_path}")
        bag = rosbag.Bag(bag_path, "w", compression="lz4")

        stamp = get_time(
            self.nusc.get(
                "ego_pose",
                self.nusc.get("sample_data", cur_sample["data"][self.lidar_channel])[
                    "ego_pose_token"
                ],
            )
        )

        if self.use_map:
            map_msg = get_scene_map(scene, nusc_map, bitmap, stamp)
            centerlines_msg = get_centerline_markers(scene, nusc_map, stamp)
            bag.write("/map", map_msg, stamp)
            bag.write("/semantic_map", centerlines_msg, stamp)
        last_map_stamp = stamp

        while cur_sample is not None:
            sample_lidar = self.nusc.get(
                "sample_data", cur_sample["data"][self.lidar_channel]
            )
            ego_pose = self.nusc.get("ego_pose", sample_lidar["ego_pose_token"])
            stamp = get_time(ego_pose)

            # write map topics every two seconds
            if self.use_map:
                if stamp - rospy.Duration(2.0) >= last_map_stamp:
                    map_msg.header.stamp = stamp
                    for marker in centerlines_msg.markers:
                        marker.header.stamp = stamp
                    bag.write("/map", map_msg, stamp)
                    bag.write("/semantic_map", centerlines_msg, stamp)
                    last_map_stamp = stamp

            # write CAN messages to /pose, /odom, and /diagnostics
            if self.use_can:
                can_msg_events = []
                for i in range(len(can_parsers)):
                    (can_msgs, index, msg_func) = can_parsers[i]
                    while index < len(can_msgs) and get_utime(can_msgs[index]) < stamp:
                        can_msg_events.append(msg_func(can_msgs[index]))
                        index += 1
                        can_parsers[i][1] = index
                can_msg_events.sort(key=lambda x: x[0])
                for msg_stamp, topic, msg in can_msg_events:
                    bag.write(topic, msg, stamp)

            # publish /tf
            tf_array = get_tfmessage(self.nusc, cur_sample, self.lidar_channel)
            bag.write("/tf", tf_array, stamp)

            # /driveable_area occupancy grid
            if self.use_map:
                write_occupancy_grid(bag, nusc_map, ego_pose, stamp)

            # iterate sensors
            for sensor_id, sample_token in cur_sample["data"].items():
                sample_data = self.nusc.get("sample_data", sample_token)
                topic = "/" + sensor_id

                # write the sensor data
                if sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(self.dataroot, sample_data, sensor_id)
                    bag.write(topic, msg, stamp)
                elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(self.dataroot, sample_data, sensor_id)
                    bag.write(topic + "/image_rect_compressed", msg, stamp)
                    msg = get_camera_info(self.nusc, sample_data, sensor_id)
                    bag.write(topic + "/camera_info", msg, stamp)

                if sample_data["sensor_modality"] == "camera":
                    msg = get_lidar_imagemarkers(
                        self.nusc, sample_lidar, sample_data, sensor_id
                    )
                    bag.write(topic + "/image_markers_lidar", msg, stamp)
                    write_boxes_imagemarkers(
                        self.nusc,
                        bag,
                        cur_sample["anns"],
                        sample_data,
                        sensor_id,
                        topic,
                        stamp,
                    )

            # publish /pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "base_link"
            pose_stamped.header.stamp = stamp
            pose_stamped.pose.orientation.w = 1
            bag.write("/pose", pose_stamped, stamp)

            # publish /gps
            coordinates = derive_latlon(location, ego_pose)
            gps = NavSatFix()
            gps.header.frame_id = "base_link"
            gps.header.stamp = stamp
            gps.status.status = 1
            gps.status.service = 1
            gps.latitude = coordinates["latitude"]
            gps.longitude = coordinates["longitude"]
            gps.altitude = get_transform(ego_pose).translation.z
            bag.write("/gps", gps, stamp)

            # publish /markers/annotations
            marker_array = MarkerArray()
            for annotation_id in cur_sample["anns"]:
                ann = self.nusc.get("sample_annotation", annotation_id)
                marker_id = int(ann["instance_token"][:4], 16)
                c = np.array(self.nusc.explorer.get_color(ann["category_name"])) / 255.0

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = stamp
                marker.id = marker_id
                marker.text = ann["instance_token"][:4]
                marker.type = Marker.CUBE
                marker.pose = get_pose(ann)
                marker.frame_locked = False  # False
                marker.scale.x = ann["size"][1]
                marker.scale.y = ann["size"][0]
                marker.scale.z = ann["size"][2]
                marker.lifetime = rospy.Duration(0.2)
                marker.color = make_color(c, 0.5)
                marker_array.markers.append(marker)
            bag.write("/markers/annotations", marker_array, stamp)

            # collect all sensor frames after this sample but before the next sample
            non_keyframe_sensor_msgs = []
            for sensor_id, sample_token in cur_sample["data"].items():
                topic = "/" + sensor_id

                next_sample_token = self.nusc.get("sample_data", sample_token)["next"]
                while next_sample_token != "":
                    next_sample_data = self.nusc.get("sample_data", next_sample_token)
                    # if next_sample_data['is_key_frame'] or get_time(next_sample_data).to_nsec() > next_stamp.to_nsec():
                    #     break
                    if next_sample_data["is_key_frame"]:
                        break

                    if next_sample_data["sensor_modality"] == "lidar":
                        msg = get_lidar(next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append(
                            (msg.header.stamp.to_nsec(), topic, msg)
                        )
                    elif next_sample_data["sensor_modality"] == "camera":
                        msg = get_camera(next_sample_data, sensor_id)
                        camera_stamp_nsec = msg.header.stamp.to_nsec()
                        non_keyframe_sensor_msgs.append(
                            (camera_stamp_nsec, topic + "/image_rect_compressed", msg)
                        )

                        msg = get_camera_info(next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append(
                            (camera_stamp_nsec, topic + "/camera_info", msg)
                        )

                        closest_lidar = find_closest_lidar(
                            cur_sample["data"][self.lidar_channel], camera_stamp_nsec
                        )
                        if closest_lidar is not None:
                            msg = get_lidar_imagemarkers(
                                closest_lidar, next_sample_data, sensor_id
                            )
                            non_keyframe_sensor_msgs.append(
                                (
                                    msg.header.stamp.to_nsec(),
                                    topic + "/image_markers_lidar",
                                    msg,
                                )
                            )
                        else:
                            msg = get_remove_imagemarkers(
                                sensor_id, self.lidar_channel, msg.header.stamp
                            )
                            non_keyframe_sensor_msgs.append(
                                (
                                    msg.header.stamp.to_nsec(),
                                    topic + "/image_markers_lidar",
                                    msg,
                                )
                            )

                        # Delete all image markers on non-keyframe camera images
                        # msg = get_remove_imagemarkers(sensor_id, 'LIDAR_TOP', msg.header.stamp)
                        # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_lidar', msg))
                        # msg = get_remove_imagemarkers(sensor_id, 'annotations', msg.header.stamp)
                        # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_annotations', msg))
                    next_sample_token = next_sample_data["next"]

            # sort and publish the non-keyframe sensor msgs
            non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
            for _, topic, msg in non_keyframe_sensor_msgs:
                bag.write(topic, msg, msg.header.stamp)

            # move to the next sample
            cur_sample = (
                self.nusc.get("sample", cur_sample["next"])
                if cur_sample.get("next") != ""
                else None
            )

        bag.close()
        print(f"Finished writing {bag_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NuScenes dataset to ROS bag.")
    parser.add_argument(
        "--scene", type=str, default="boston-seaport", help="scene name"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-mini", help="dataset version"
    )
    parser.add_argument(
        "--dataroot", type=str, default="./data", help="dataset root directory"
    )
    parser.add_argument(
        "--lidar_channel", type=str, default="lidar-fusion", help="lidar channel"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    nuscenes2bag = Nuscenes2Bag(
        scene=args.scene,
        version=args.version,
        dataroot=args.dataroot,
        lidar_channel=args.lidar_channel,
        use_map=False,
        use_can=False,
    )
    nuscenes2bag.convert()


if __name__ == "__main__":
    main()
