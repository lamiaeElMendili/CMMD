
import numpy as np
import open3d as o3d
import argparse
from laserscan import LaserScan, SemLaserScan
from laserscanvis import LaserScanVis
import os
import yaml



color_maps = {
    'semanticposs' : {
                        0: (250, 178, 50),
                        1: (255, 196, 0),
                        2: (25, 25, 255),
                        3: (107, 98, 56),
                        4: (157, 234, 50),
                        5: (173, 23, 121),
                        }
}






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        help="Path to dataset folder")
    
    parser.add_argument("--dataset_name",
                        type=str,
                        help="Path to dataset folder")


    parser.add_argument("--predictions",
                        type=str,
                        help="Path to predictions folder",
                        default=None,
                        required=False)
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))
    
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))

    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print(f"Sequence folder {scan_paths} exists! Using sequence from {scan_paths}")
    else:
        print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
        quit()


    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    if FLAGS.predictions is not None:
      label_paths = os.path.join(FLAGS.predictions, "sequences",
                                 FLAGS.sequence, "predictions")
    else:
      label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
      

    if os.path.isdir(label_paths):
      print(f"Labels folder {label_paths} exists! Using labels from {label_paths}")
    else:
      print(f"Labels folder {label_paths} doesn't exist! Exiting...")
      quit()

    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()


    color_dict = CFG["color_map"]


    scan = SemLaserScan(color_dict, project=True)


    vis = LaserScanVis(scan=scan,
                        scan_names=scan_names,
                        label_names=label_names,
                        offset=0,
                        semantics=True, instances=False, images=False, link=False)


    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()




def visualize_pcd_clusters(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])




