import os

from config import gels, gelParams, dt, taskSettings, objs

if __name__ == "__main__":
    tasks = [
        "press", 
        # "shear", 
        # "rotate",
    ]

    os.makedirs("output/logs", exist_ok=True)

    for task in tasks:
        gel = gels[task]
        taskSetting = taskSettings[task]
        for objName, objPth in objs[task].items():
            exp_name = f"{task}_{objName}_{gel['name']}_dt={dt}_fricMu={taskSetting['fricMu']}_pressSteps={taskSetting['pressSteps']}_pressDepth={taskSetting['pressDepth']}_pressVel={taskSetting['pressVel']}_taskSteps={taskSetting['taskSteps']}_moveVel={taskSetting['moveVel']}"
            print(f"Running experiment: {exp_name}")
            command = (
                f"stdbuf -o0 ./build/main "
                f"{exp_name} "
                f"false "
                f"true " # cg solver
                # f"false " # host solver
                f"1e-20 "
                f"10 "
                f"true "
                f"{gel['pth']} "
                f"{objPth} "
                f"{gel['isBCPth']} "
                f"{taskSetting['fricMu']} "
                f"{task} "
                f"{taskSetting['pressSteps']} "
                f"{taskSetting['pressDepth']} "
                f"{taskSetting['pressVel']} "
                f"{taskSetting['taskSteps']} "
                f"{taskSetting['moveVel']} "
                f"{dt} "
                f"> output/logs/{exp_name}.log"
            )
            print(f"\tcommand: {command}")
            os.system(command)
            

