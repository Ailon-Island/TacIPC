# TacIPC: Intersection- and Inversion-Free FEM-Based Elastomer Simulation for Optical Tactile Sensors
## Requirements
- CMake >= 3.20
- gcc/g++ >= 10
- SuiteSparse = 7.3.1
- xrandr
- xinerama
- xcursor
- xi
- OpenGL
```sh
# install with apt
sudo apt install libsuitesparse-dev
sudo apt install libxrandr-dev
sudo apt install libxinerama-dev 
sudo apt install libxcursor-dev
sudo apt install libxi-dev
sudo apt install libgl1-mesa-dev
```

## Configure && Build
Build the project, and the executable will be generated in the `build` directory.
```sh
cmake -Bbuild build # configure
cmake --build build --parallel [N_threads] # build
```

## Run example experiments
### Run with Python script (headless by default)
1.  Run the experiment
    1. Run experiment script
        ```sh
        python scripts/run_experiments.py
        ```
    2. Check out the experiment result directory `output/experiments/[timestamp]`, and the log files in `output/logs/[expName].log`.
        - `output/experiments/[timestamp]/config.json` contains the experiment configuration.
        - `output/experiments/[timestamp]/objs/` contains 
            - the deformed object meshes (`[objName]_[frameId].obj`)
            - the object state files (`[objName]_[frameId].json`)
2.  Postprocess and obtain marker displacements

    1.  Create `output/experiments_markers` directory if not exists.
    2.  Move/Copy the result directories into `output/experiments_markers`.
    3.  Run the postprocess script
        ```sh
        python process/gen_marker_disp.py
        ```
    4.  Check out the marker displacement files in `output/experiments_markers/[expName]/markers/`.
5.  (optional) Visualize the marker displacements
    1.  Run the visualization script
        ```sh
        python process/vis_marker_disp.py
        ```
    2.  Check out the visualization in `output/experiments_markers/[expName]/marker.mp4`.
### Run directly in command line
1.  Run the experiment
    ```sh
    stdbuf -o0 ./build/main \
        press_example \
        true \
        true \
        1e-20 \
        10 \
        true \
        resources/gel/gelslim-gel-l_0.1/gelslim-gel-l_0.1.msh \
        resources/press_obj/Wave_8134.obj \
        resources/gel/gelslim-gel-l_0.1/isBC_gelslim-gel-l_0.1.json \
        0 \
        press \
        6 \
        1 \
        25 \
        0 \
        25 \
        0.01 \
        > output/logs/press_example.log
    ```
    Or generally, run customized experiments following
    ```sh
    stdbuf -o0 ./build/main \
        [expName] \
        [enableGui] \
        [enableCGSolver] \
        [cgRel] \
        [PNCap] \
        [enableInversionPrevention] \
        [gelPth] \
        [objPth] \
        [isBCPth] \
        [fricMu] \
        [moveType] \
        [pressSteps] \
        [pressDepth] \
        [pressVel] \
        [taskSteps] \
        [moveVel] \
        [dt] \
        > [logPth]
    ```
2. Postprocess and obtain marker displacements (same as above)