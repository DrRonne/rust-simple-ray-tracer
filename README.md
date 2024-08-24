# Rust simple ray tracer

This is an experiment I started to learn writing code in rust as well as brush
up a bit of GPU programming (OpenCL).

At the current state, this is by no means an optimized or finished engine.
Currently, it can draw spheres and interact with a singular directional light.
It will also cast shadows. It doesn't really do any light bounces at the moment, it just casts 1 ray per pixel and calculates what color to render for that pixel
based on the interaction with the directional light. It does check if nothing
is in the way to that directional light, meaning that it does cast shadows.

I also provided a janky way to control the camera using WASD to move the camera
and rotate the camera using the mouse (while the left button is pressed).

This is just a POC at the moment and I didn't take much into consideration to
make it pretty or follow any standards at all really. I started this project
without knowing much about rust.