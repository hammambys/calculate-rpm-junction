# Event Camera Dataset

The dataset for the event camera challenge includes event data of the following scenarios:

### Rotating Fan
Under the `fan` folder you find three distinct scenarios: 
-  `fan_const_rpm` is event data produced by a fan rotating at constant speed, duration ~ 10 s, rpm ~ 1100
- `fan_varying_rpm` is event data produced by a fan which changes its rotating speed during the clip, duration ~ 20 s, rpm $\in$ [1100, 1300]
- `fan_varying_rpm_turning` is event data produced by a fan which changes its rotating speed and orientation w.r.t the camera during the clip, duration ~ 25 s, rpm $\in$ [1100, 1300]

### Drone Idle
`drone_idle` is event data recorded from a drone hovering stationary at roughly 100 m away from the camera with a tree wobbling on the background, duration ~ 10 s, rpm $\in$ [5000, 6000]

### Drone Moving
`drone_moving` is event data recorded from a drone moving around at roughly 100 m away from the camera with a wobbling tree and a plane on the background and, duration ~ 20 s, rpm $\in$ [5500, 6500]

### FRED 0 & 1
These samples are taken from the larger [FRED](https://miccunifi.github.io/FRED/) dataset which includes event data and normal video footage of drones flying on various conditions. Since the samples include annotated trajectiories they can be helpful in building a solution for drone tracking. You can convert the .raw event data to .dat format using the conversion scripts in [Metavision SDK](https://docs.prophesee.ai/stable/installation/index.html).
