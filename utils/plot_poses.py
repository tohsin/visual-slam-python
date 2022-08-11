# https://github.com/stevenlovegrove/Pangolin/tree/master/examples/HelloPangolin

import OpenGL.GL as gl
import pangolin

import numpy as np



def display_poses(poses):
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    
    trajectory = [[0, -6, 6]]
    for i in range(300):
        trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
    trajectory = np.array(trajectory)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
     
        # Draw camera
        for pose in poses:
            # pose = np.identity(4)
            # pose[:3, 3] = np.random.randn(3)
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)


        pangolin.FinishFrame()


