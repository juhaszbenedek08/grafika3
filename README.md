# grafika3 - Rubber Sheet

## Specification

Create rubber sheet simulator to demonstrate gravity.

The rubber sheet has a flat torus topology (what goes out comes in on the opposite side)
The rubber sheet is has diffuse and ambient coefficients that darkens in steps according to the indentation.

Place a high mass body by pressing the right mouse button.
The high mass bodies curve the rubber sheet.
The high mass bodies are invisible.

Place a small ball by pressing the left mouse button.
The ball is placed in the bottom left corner.
The ball has an initial momentum with direction and amplitude depending on the mouse position.
The balls move frictionlessly.
The balls are coloured diffuse-specular.

The balls are absorved if colliding with the masses.
The collisions between the balls need not be considered.

Two point light sources illumiate the sheet.
The light soruces rotate around each other's initial position, according to the quaternion:
q=[ cos(t/4), sin(t/4)cos(t)/2, sin(t/4)sin(t)/2, √(3/4)sin(t/4)].

The rubber sheet is viewed from above.
Pressing SPACE causes our virtual camera to stick to the first ball not yet absorbed, so we can follow its point of view.
If no ball is alive, the camera returns to initial position.

# Result

![Balls](https://user-images.githubusercontent.com/59647190/161544503-67fb32cf-7215-4d3a-9b51-64ae66228a9e.png)
