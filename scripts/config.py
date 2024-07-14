pressGel = {
    "name"      : "gel_0.1",
    "pth"       : "resources/gel/gelslim-gel-l_0.1/gelslim-gel-l_0.1.msh",
    "isBCPth"   : "resources/gel/gelslim-gel-l_0.1/isBC_gelslim-gel-l_0.1.json"
}
fricGel = {
    "name"      : "gel_0.1-3mm",
    "pth"       : "resources/gel/gelslim-gel-l_0.1-3mm/gelslim-gel-l_0.1-3mm.msh",
    "isBCPth"   : "resources/gel/gelslim-gel-l_0.1-3mm/isBC_gelslim-gel-l_0.1-3mm.json"
}
gels = {
    "press" : pressGel,
    "shear"  : fricGel,
    "rotate" : fricGel,
}

gelParams = {
    "density"   : 1.01e-3,  # g / mm^3
    "E"         : 1.23e5,   # g / (mm * s^2) 
    "nu"        : 0.43,    
}

dt = 0.01

pressSettings = {
    "fricMu"        : 0,  
    "pressSteps"    : 6,   
    "pressDepth"    : 1,    # mm
    "pressVel"      : 25,   # mm/s
    "taskSteps"     : 0,   
    "moveVel"       : 25,   # mm/s       
}
shearSettings = {
    "fricMu"        : 0.7,  
    "pressSteps"    : 4,   
    "pressDepth"    : 0.5,  # mm
    "pressVel"      : 25,   # mm/s
    "taskSteps"     : (taskSteps := 100),
    "moveVel"       : 1 / (taskSteps * dt),    # mm/s       
}
rotateSettings = {
    "fricMu"        : 0.7,      
    "pressSteps"    : 4,   
    "pressDepth"    : 0.5,      # mm
    "pressVel"      : 25,       # mm/s
    "taskSteps"     : (taskSteps := 200),
    "moveVel"       : 0.25 / (taskSteps * dt),    # rad/s       
}
taskSettings = {
    "press"     : pressSettings,
    "shear"     : shearSettings,
    "rotate"    : rotateSettings,
}

pressObjs = {
    # "board1": "resources/press_obj/board1.obj",
    # "board2": "resources/press_obj/board2.obj",
    # "board3": "resources/press_obj/board3.obj",
    # "board4": "resources/press_obj/board4.obj",
    # "board5": "resources/press_obj/board5.obj",
    # "board6": "resources/press_obj/board6.obj",
    # "board7": "resources/press_obj/board7.obj",
    # "board8": "resources/press_obj/board8.obj",
    # "board9": "resources/press_obj/board9.obj",
    # "board10": "resources/press_obj/board10.obj",
    # "Anubis": "resources/press_obj/Anubis_6607.obj",
    # "Bear": "resources/press_obj/Bear_8028.obj",
    # "Chicken": "resources/press_obj/Chicken_8210.obj",
    # "Dragon1": "resources/press_obj/dragon_01.obj",
    # "Dragon2": "resources/press_obj/dragon_02.obj",
    # "Dragon3": "resources/press_obj/dragon_03.obj",
    # "Dragon4": "resources/press_obj/dragon_04.obj",
    # "LinoCake": "resources/press_obj/LinoCake_8051.obj",
    # "Lotus": "resources/press_obj/Lotus.obj",
    "Wave": "resources/press_obj/Wave_8134.obj",
    # "coin-2": "resources/press_obj/coin_5002.obj",
}
fricObjs = {
    "board3": "resources/press_obj/board3.obj",
    # "pen": "resources/fric_obj/pen.obj",
    # "wood-block": "resources/fric_obj/wood_block.obj",
}
objs = {
    "press" : pressObjs,
    "shear" : fricObjs,
    "rotate": fricObjs,
}