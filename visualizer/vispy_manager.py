#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
from abc import ABC, abstractmethod
import numpy as np
from vispy.scene.widgets.colorbar import ColorBarWidget

class VispyManager(ABC):
  def __init__(self, offset, total, images, instances):
    self.canvas, self.grid = self.add_canvas('interactive', 'scan')
    self.offset = offset
    self.images = images
    self.instances = instances
    self.n_images = 2
    self.img_canvas_W = 1024
    self.img_canvas_H = 64 * self.n_images
    self.img_canvas = None
    if self.images:
      self.img_canvas, self.img_grid = self.add_canvas('interactive', 'img', size=(self.img_canvas_W, self.img_canvas_H))

    self.total = total

  def add_canvas(self, keys, title, size=None):
    canvas = None
    if size:
      canvas = SceneCanvas(keys=keys, bgcolor='w', show=True, size=size, title=title)
    else:
      canvas = SceneCanvas(keys=keys, bgcolor='w', show=True, title=title)





    canvas.events.key_press.connect(self.key_press)
    canvas.events.draw.connect(self.draw)
    grid = canvas.central_widget.add_grid()
    return canvas, grid

  @staticmethod
  def block_key_press(canvas):
    canvas.events.key_press.block()

  @staticmethod
  def key_press_blocked(canvas):
    return canvas.events.key_press.blocked()

  @staticmethod
  def key_press_unblocked(canvas):
    return not canvas.events.key_press.blocked()

  @staticmethod
  def unblock_key_press(canvas):
    canvas.events.key_press.unblock()

  def add_viewbox(self, row, col, border_color='white'):
    view = vispy.scene.widgets.ViewBox(border_color=border_color, parent=self.canvas.scene)
    self.grid.add_widget(view, row, col)
    vis = visuals.Markers()
    view.camera = 'turntable'
    view.add(vis)
    visuals.XYZAxis(parent=view.scene)
    return view, vis

  def add_image_viewbox(self, row, col, border_color='white'):
    img_view = vispy.scene.widgets.ViewBox(
      border_color=border_color, parent=self.img_canvas.scene)
    self.img_grid.add_widget(img_view, row, col)
    img_vis = visuals.Image(cmap='viridis')
    img_view.add(img_vis)
    return img_view, img_vis

  def key_press(self, event):
    VispyManager.block_key_press(self.canvas)
    if self.img_canvas:
      VispyManager.block_key_press(self.img_canvas)
    if event.key == 'N':
      self.offset += 1
      self.offset %= self.total
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      self.offset %= self.total
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

    elif event.key == 'R':
        # Reset the view
        for view in self.views:
            view.camera.reset()

    elif event.key == 'M':
      # Match all views with the first view
      self.match_views()
    elif event.key == 'Left':
        self.pan_camera(-1, 0)  # Pan left
    elif event.key == 'Right':
        self.pan_camera(1, 0)  # Pan right
    elif event.key == 'Up':
        self.pan_camera(0, -1)  # Pan up
    elif event.key == 'Down':
        self.pan_camera(0, 1)  # Pan down

  def pan_camera(self, x, y):
      for view in self.views:
          # Adjust the camera center to pan
          view.camera.center = (
              view.camera.center[0] + x, 
              view.camera.center[1] + y, 
              view.camera.center[2])

  def destroy(self):
    if self.canvas:
      self.canvas.close()
    if self.img_canvas:
      self.img_canvas.close()
    vispy.app.quit()

  def draw(self, event):
    if VispyManager.key_press_blocked(self.canvas):
      VispyManager.unblock_key_press(self.canvas)
    if self.img_canvas and VispyManager.key_press_blocked(self.img_canvas):
      VispyManager.unblock_key_press(self.img_canvas)

  def run(self):
    vispy.app.run()

  @abstractmethod
  def update_scan(self):
    raise NotImplementedError

  @abstractmethod
  def reset(self):
    raise NotImplementedError
  

  def match_views(self):
      """Make all views match the camera settings of the first view"""
      print(self.views)
      if len(self.views) > 1:
          first_view_camera = self.views[0].camera
          for view in self.views[1:]:
              # Copy camera parameters from the first view
              view.camera.azimuth = first_view_camera.azimuth
              view.camera.elevation = first_view_camera.elevation
              view.camera.distance = first_view_camera.distance
              view.camera.center = first_view_camera.center
              view.camera.scale_factor = first_view_camera.scale_factor



