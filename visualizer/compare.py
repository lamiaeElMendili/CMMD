
import numpy as np
import open3d as o3d
import argparse
from laserscan import LaserScan, SemLaserScan
from laserscancomp import LaserScanComp
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt


color_maps = {
    'semanticposs' : {  -1: (0, 0, 0),
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
                        required=False,
                        nargs='+')

    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )


    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    print("Opening config file %s" % FLAGS.config)
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


    
    
    labels_b = FLAGS.predictions[0]
    labels_c = FLAGS.predictions[1]
    labels_d = FLAGS.predictions[2]


    label_a_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "labels")

    label_b_paths = os.path.join(labels_b, "sequences",
                                 FLAGS.sequence, "predictions")
    

    label_c_paths = os.path.join(labels_c, "sequences",
                                 FLAGS.sequence, "predictions")

    label_d_paths = os.path.join(labels_d, "sequences",
                                 FLAGS.sequence, "predictions")





    if os.path.isdir(label_a_paths):
        print("Labels folder a exists! Using labels from %s" % label_a_paths)
    else:
        print("Labels folder a doesn't exist! Exiting...")
        quit()


    print(label_b_paths)
    if os.path.isdir(label_b_paths):
        print("Labels folder b exists! Using labels from %s" % label_b_paths)
    else:
        print("Labels folder b doesn't exist! Exiting...")
        quit()


    if os.path.isdir(label_c_paths):
        print("Labels folder c exists! Using labels from %s" % label_c_paths)
    else:
        print("Labels folder c doesn't exist! Exiting...")
        quit()
    
    if os.path.isdir(label_d_paths):
        print("Labels folder d exists! Using labels from %s" % label_d_paths)
    
    else:
        print("Labels folder d doesn't exist! Exiting...")
        quit()

    

    

    # populate the pointclouds
    label_a_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_a_paths)) for f in fn]
    label_a_names.sort()
    label_b_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_b_paths)) for f in fn]
    label_b_names.sort()
    label_c_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_c_paths)) for f in fn]
    label_c_names.sort()
    label_d_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_d_paths)) for f in fn]
    label_d_names.sort()

    #print(label_d_names[3])
    #print('*****************')


    color_dict = CFG["color_map"]

    learning_map_inv = CFG["learning_map_inv"]
    learning_map = CFG["learning_map"]
    color_dict = {key: color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items() if key in learning_map_inv.values()}

    # Sort the color map by class ID
    sorted_color_map = sorted(color_dict.items())
    normalized_color_map = {class_id: [comp / 255 for comp in color] for class_id, color in color_dict.items()}


    # Convert color map to a list of colors
    colors = [normalized_color_map[key] for key, _ in sorted_color_map]



    # Create a seaborn color palette
    palette = sns.color_palette(colors, as_cmap=True)

    # Plot rectangles with class names
    fig, ax = plt.subplots(figsize=(20, 2))
    for i, (class_id, color) in enumerate(sorted_color_map):
        rect = plt.Rectangle((i * 1.1, 0.5), 1, 0.8, color=normalized_color_map[class_id])
        ax.add_patch(rect)
        ax.text(i * 1.1 + 0.5, 0, f'{CFG["labels"][class_id]}', verticalalignment='bottom', horizontalalignment='center', rotation=45)

    # Customize plot
    ax.set_xlim(-0.5, len(color_dict) * 1.1 - 0.5)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    plt.show()








    #color_dict = color_maps[FLAGS.dataset_name]


    scan_b = SemLaserScan(color_dict, project=True)
    scan_a = SemLaserScan(color_dict, project=True)
    scan_c = SemLaserScan(color_dict, project=True)
    scan_d = SemLaserScan(color_dict, project=True)



    vis = LaserScanComp(scans=(scan_a, scan_b, scan_c, scan_d),
                        scan_names=scan_names,
                        label_names=(label_a_names, label_b_names, label_c_names, label_d_names),
                        offset=0,instances=False, images=False, link=False)


    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\r: reset view")
    print('\m: match views to first view')
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()






