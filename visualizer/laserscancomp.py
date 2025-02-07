#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from vispy_manager import VispyManager
import numpy as np

from vispy.scene.widgets import ColorBarWidget

class LaserScanComp(VispyManager):
  """Class that creates and handles a side-by-side pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, instances=False, link=False):
    super().__init__(offset, len(scan_names), images, instances)

    self.scan_a_view = None
    self.scan_a_vis = None
    self.scan_b_view = None
    self.scan_b_vis = None
    self.inst_a_view = None
    self.inst_a_vis = None
    self.inst_b_view = None
    self.inst_b_vis = None
    self.img_a_view = None
    self.img_a_vis = None
    self.img_b_view = None
    self.img_b_vis = None
    self.img_inst_a_view = None
    self.img_inst_a_vis = None
    self.img_inst_b_view = None
    self.img_inst_b_vis = None

    self.scans = scans

    #self.scan_a, self.scan_b = scans
    self.scan_names = scan_names
    #self.label_a_names, self.label_b_names = label_names
    self.label_names = label_names
    self.link = link
    self.reset()
    self.update_scan()

  def reset(self):
      n = len(self.scans)
      """Prepares the canvas(es) for the visualizer, with a given number of scans"""
      self.views = []
      self.visuals = []
      
      # Create viewboxes for each scan
      for i in range(n):
          # Determine the grid coordinates for each viewbox
          x = i // 2  # rows (divided by 2 for a simple layout)
          y = i % 2   # columns (split across 2 columns)
          
          view, vis = super().add_viewbox(x, y)
          self.views.append(view)
          self.visuals.append(vis)

      # Link cameras if needed
      if self.link and n > 1:
          # Link all cameras in a simple chain
          for i in range(1, len(self.views)):
              self.views[i - 1].camera.link(self.views[i].camera)
          
      # Handle image viewboxes if self.images is set
      if self.images:
          self.img_views = []
          self.img_visuals = []
          
          for i in range(2):  # Assuming you need 2 image viewboxes
              view, vis = super().add_image_viewbox(i, 0)
              self.img_views.append(view)
              self.img_visuals.append(vis)
          
          if self.instances and len(self.img_views) > 0:
              # Adding more instance viewboxes
              inst_view, inst_vis = super().add_image_viewbox(2, 0)
              self.img_views.append(inst_view)
              self.img_visuals.append(inst_vis)

      # Handle instance viewboxes if self.instances is set
      if self.instances:
          self.inst_views = []
          self.inst_visuals = []
          
          for i in range(2):  # Assuming two instance viewboxes
              x = 1  # Row 1
              y = i  # Column based on index
              view, vis = super().add_viewbox(x, y)
              self.inst_views.append(view)
              self.inst_visuals.append(vis)
          
          # Link cameras in the instance viewboxes if needed
          if self.link:
              if len(self.inst_views) > 0:
                  self.views[0].camera.link(self.inst_views[0].camera)
                  if len(self.inst_views) > 1:
                      self.inst_views[0].camera.link(self.inst_views[1].camera)



        

  def update_scan(self):

    for (i, scan) in enumerate(self.scans) :
        scan.open_scan(self.scan_names[self.offset])
        scan.open_label(self.label_names[i][self.offset])
        scan.colorize()
        self.visuals[i].set_data(scan.points,
                            face_color=scan.sem_label_color[..., ::-1],
                            edge_color=scan.sem_label_color[..., ::-1],
                            size=1)
      
   



        if self.instances:

            self.inst_visuals[i].set_data(scan.points,
                                    face_color=scan.inst_label_color[..., ::-1],
                                    edge_color=scan.inst_label_color[..., ::-1],
                                    size=1)


      #if self.images:
      #  self.img_a_vis.set_data(self.scan_a.proj_sem_color[..., ::-1])
      #  self.img_a_vis.update()
      #  self.img_b_vis.set_data(self.scan_b.proj_sem_color[..., ::-1])
      #  self.img_b_vis.update()

      #  if self.instances:
      #    self.img_inst_a_vis.set_data(self.scan_a.proj_inst_color[..., ::-1])
      #    self.img_inst_a_vis.update()
      #    self.img_inst_b_vis.set_data(self.scan_b.proj_inst_color[..., ::-1])
      #    self.img_inst_b_vis.update()