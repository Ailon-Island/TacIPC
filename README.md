# TacIPC: Intersection- and Inversion-Free FEM-Based Elastomer Simulation for Optical Tactile Sensors
<p align="center">
  <a href='https://ieeexplore.ieee.org/abstract/document/10410895/'>
    <img src='https://img.shields.io/badge/Paper-PDF-blue?style=flat&logo=Googlescholar&logoColor=blue' alt='Paper PDF'>
  </a>
</p>

## Requirements
- CMake >= 3.20
- gcc/g++ >= 10
- SuiteSparse = 7.3.1
- xrandr
- xinerama
- xcursor
- xi
- OpenGL
- zlib
```sh
# install with apt
sudo apt install libsuitesparse-dev
sudo apt install libxrandr-dev
sudo apt install libxinerama-dev 
sudo apt install libxcursor-dev
sudo apt install libxi-dev
sudo apt install libgl1-mesa-dev
sudo apt install zlib1g-dev
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
2. Postprocess and obtain marker displacements (same as above)

## Run custom experiments
-   Run experiment with custom configurations
    
    Modify `scripts/config.py` and `run_experiments.py`, or just run command following
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

-   Postprocess custom tactile sensor gel
    1.  Prepare your tactile sensor gel mesh and boundary condition file.
        -   The gel mesh should be tetrahedral and in `.msh` format. The length should be in milimeters.
        -   The boundary condition file should be in `.json` format. It is a list of binary numbers, where `1` indicates the corresponding node is fixed, and `0` indicates the node is free to move.
    2.  Take `gel_0.obj` after experiment as surface mesh(recommended), or extract it mannually.
    3.  Generate the triangular barycentric weights on the surface mesh for markers, and save as `marker_bc_ws.pkl` in the same directory as your `.msh` gel mesh. You may refer to `process/gen_markers.py`.
    4.  Postprocess as usual.

-   Custom tasks

    Refer to `main.cu` to implement your own tasks.
    -   Program entrance
        -   Modify arguments in `main()`
    -   Task implementation (in `prepare()`)
        -   Modify configuration to be dumped in `config`
        -   Modify solver settings
        -   Modify `move` lambda to implement your own task (usually changing the target state a bit each step)

## Cite this work
If you use TacIPC in your work, please cite us.
```bibtex
@article{tacipc2024,
  title={TacIPC: Intersection-and Inversion-free FEM-based Elastomer Simulation For Optical Tactile Sensors},
  author={Du, Wenxin and Xu, Wenqiang and Ren, Jieji and Yu, Zhenjun and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```
