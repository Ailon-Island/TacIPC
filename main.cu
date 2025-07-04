#include <filesystem>
#include <fstream>
#include <iostream>
#include <tacipc/debug.hpp>
#include <tacipc/generation.cuh>
#include <tacipc/meshio/readMesh.hpp>
#include <tacipc/meta.hpp>
#include <tacipc/types.hpp>
#include <tacipc/serialization.cuh>
#include <tacipc/solver/Solver.cuh>
#include <tacipc/viewer/viewer.cuh>

namespace fs = std::filesystem;
using T = tacipc::ABDSolver::T;

std::string getTimestamp() {
    auto currentTimePoint = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(currentTimePoint);
    std::tm* timeInfo = std::localtime(&currentTime);
    std::ostringstream oss;
    oss << std::put_time(timeInfo, "%Y-%m-%d_%H:%M:%S");
    return oss.str();
}

template <class MatT> static void printMat(const MatT &mat)
{
    for (int i = 0; i < MatT::template range_t<0>::value; i++)
    {
        for (int j = 0; j < MatT::template range_t<1>::value; j++)
            fmt::print("{:.4f},\t", mat(i, j));
        fmt::print("\n");
    }
}

template <class VecT> static void printVec(const VecT &vec)
{
    for (int i = 0; i < VecT::template range_t<0>::value; i++)
        fmt::print("{:4f},\t", vec(i));
    fmt::print("\n");
}

Eigen::Matrix3d vectorsToRotationMatrix(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    Eigen::Vector3d axis = v1.cross(v2).normalized();
    auto angle = std::acos(v1.dot(v2) / (v1.norm() * v2.norm()));
    
    Eigen::AngleAxisd rotation(angle, axis);
    Eigen::Matrix3d rotationMatrix = rotation.toRotationMatrix();
    
    return rotationMatrix;
}

auto prepare(std::string expName, bool enableCG, T cgRel, int PNCap, bool enableInversionPrevention, std::string gelPth, std::string objPth, std::string isBCPth, T fricMu, std::string moveType, int pressSteps = 6, T pressDepth = 1., T pressVel = 25, int taskSteps = 0, T moveVel = 1., T dt = 0.01)
{
    using vec3 = tacipc::ABDSolver::vec3;
    using mat4 = tacipc::ABDSolver::mat4;

    using RigidTriBody = tacipc::RigidTriangleBody<T>;
    using RigidTetBody = tacipc::RigidTetBody<T>;
    using SoftTriBody = tacipc::SoftTriangleBody<T>;
    using SoftTetBody = tacipc::SoftTetBody<T>;
    using TriRigidBodyProperties = tacipc::RigidBodyProperties<T>;
    using TetRigidBodyProperties = tacipc::RigidBodyProperties<T>;

    auto timestamp = getTimestamp();
    auto expDir = fmt::format("output/experiments/{}", timestamp);
    auto objsDir = fmt::format("{}/objs", expDir);

    fs::create_directories(expDir);
    fs::create_directories(objsDir);

    tacipc::Logger logger{fmt::format("{}/log.txt", expDir)};
    logger.log("[EXP] {}\n", expName);
    logger.log("[START] experiment starts at {}\n", timestamp);

    json config;
    config["exp_name"] = expName;
    config["gel_pth"] = gelPth;
    config["obj_pth"] = objPth;
    config["isBC_pth"] = isBCPth;
    config["fric_mu"] = fricMu;
    config["move_type"] = moveType;
    config["press_steps"] = pressSteps;
    config["press_depth"] = pressDepth;
    config["press_vel"] = pressVel;
    config["task_steps"] = taskSteps;
    config["move_vel"] = moveVel;
    config["dt"] = dt;
    config["enable_cg"] = enableCG;
    config["cg_rel"] = cgRel;
    config["PN_cap"] = PNCap;
    config["inversion_free"] = enableInversionPrevention;
    config["start time"] = timestamp;
    {
        std::ofstream ofs(fmt::format("{}/config.json", expDir));
        ofs << config.dump(4);
    }

    // T dt = 0.01;
    int frameSubsteps = 1;
    // bool enableCG = true;
    // bool enableInversionPrevention = true;
    bool enablePureRigidOpt = true;
    bool enableGround = false;
    bool enableContact = true;
    bool enableMollification = true; 
    bool enableContactEE = true;
    bool enableFriction = fricMu > 0;
    bool enableBoundaryFriction = false;
    bool enableSoftBC = false;
    std::size_t layerCps = 64;
    std::size_t estNumCps =     20000001;
    std::size_t dynHessCps =     3000002;
    T kinematicALCoef = 1e6;
    T pnRel = 2e-2;
    // T cgRel = 1e-3;
    int fricIterCap = 2;
    // int PNCap = 30;
    int CGCap = 1000;
    int CCDCap = 20000;
    T kappa0 = 1e6;
    bool useAbsKappaDhat = false;
    // T fricMu = 0.7;
    T boundaryKappa = kappa0;
    T springStiffness = 1e4; 
    T abdSpringStiffness = 1e4; 
    T xi = 0;
    T dHat = 1e-3;
    T epsv = 0;
    T kinematicALTol = 1e-1;
    T consTol = 1e-2;
    T armijoParam = 1e-5;
    vec3 groundNormal{0, 1, 0};
    eigen::vec3d gn0{0, 1, 0}, gn{groundNormal[0], groundNormal[1], groundNormal[2]};
    Eigen::Matrix3d groundRot = vectorsToRotationMatrix(gn0, gn);
    T gravity = 0; 

    T pressLength = pressSteps * pressVel * dt;
    T dist = pressLength - pressDepth;

    // auto groundMesh = meshio::readObjFile("resources/ground.obj");
    auto gelMesh = meshio::readMshFile(gelPth);
    auto objMesh = meshio::readObjFile(objPth);

    logger.log("gel mesh file name: {}\n", gelPth);
    logger.log("obj mesh file name: {}\n", objPth);
    logger.log("friction coefficient: {}\n", fricMu);

    tacipc::Transform<T, 3> gelTrans{};
    {
        gelTrans.setToScale(vec3::constant(1e3));
    }

    tacipc::Transform<T, 3> objTrans{};
    {
        vec3 offset {0, 0, 
            gelMesh.getVerts().col(2).minCoeff() -
            objMesh.getVerts().col(2).maxCoeff() - dist};
        objTrans.setToTranslation(offset);
    }

    auto gel = tacipc::genTetBodyFromTetMesh<SoftTetBody>(gelMesh, 1.01e-3, gelTrans);
    auto obj = tacipc::genTriBodyFromTriMesh<RigidTriBody>(objMesh, 1e-3, {objTrans});
    
    tacipc::setConstitutiveModel<tacipc::StVKWithHenckyModel>(*gel, 1.23e5, 0.43);
    tacipc::setConstitutiveModel<tacipc::OrthogonalModel>(*obj, 3e8);
    
    {
        obj->bodyTypeProperties.setBC();
        Eigen::VectorX<bool> gelIsBC; 
        {
            json j = json::parse(std::fstream(isBCPth));
            j.get_to(gelIsBC);
        }
        gel->bodyTypeProperties.setBC(gelIsBC);
    }

    fmt::print("objects ready\n");

    {
        gel->layer = 1;
    }
    
    tacipc::CollisionMatrix<> collisionMat{layerCps};
    {
        collisionMat.setCollision(1, 1, false); // gel-gel
        // collisionMat.setCollision(1, 0, false); // gel-obj
    }

    fmt::print("collision matrix ready\n");


    std::vector<tacipc::BodySP<T>> bodies{gel, obj};
    
    auto solver = std::make_shared<tacipc::ABDSolver>(bodies, collisionMat, dt, frameSubsteps, enableCG, 
                                enableInversionPrevention, 
                                enablePureRigidOpt, 
                                  enableGround, enableContact, enableMollification, 
                                  enableContactEE,
                                  enableFriction, enableBoundaryFriction, enableSoftBC, 
                                  layerCps, estNumCps, dynHessCps, kinematicALCoef, pnRel, cgRel, 
                                  fricIterCap, PNCap, CGCap, CCDCap, kappa0, useAbsKappaDhat, fricMu, boundaryKappa, 
                                  springStiffness, abdSpringStiffness,
                                  xi, dHat, epsv, kinematicALTol, consTol, armijoParam, 
                                  groundNormal, gravity);
    
    fmt::print("solver ready\n");

    int totalSteps = pressSteps + taskSteps;
    tacipc::Transform<T, 3> pressMotion{};
    {
        pressMotion.setToTranslation(vec3{0, 0, pressVel * dt});
    }
    tacipc::Transform<T, 3> taskMotion{};
    {
        if (moveType == "press")
            taskMotion.setToTranslation(vec3::zeros());
        else if (moveType == "shear")
            taskMotion.setToTranslation(vec3{0, -moveVel * dt, 0});
        else if (moveType == "rotate")
        {
            tacipc::Rotation<T, 3> rot{vec3{0, 0, 1}, -moveVel * dt};
            taskMotion.setToRotation(rot);
        }
    }
    auto move = [pressSteps, totalSteps, pressMotion, taskMotion, obj](mat4 &objMotion, int step) mutable 
    {
        if (step <= pressSteps)
        {
            objMotion = pressMotion * objMotion;
            fmt::print("after press motion\n");
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                    fmt::print("{:.4f},\t", objMotion(i, j));
                fmt::print("\n");
            }
            obj->bodyTypeProperties.setBCTarget(objMotion);
        }
        else if (step <= totalSteps)
        {
            objMotion = taskMotion * objMotion;
            obj->bodyTypeProperties.setBCTarget(objMotion);
        }
        else 
            return false; // quit flag

        return true;
    };

    auto save_scene = [=](int step)
    {
        {
            auto pth1 = fmt::format("{}/obj_{}.obj", objsDir, step);
            auto pth2 = fmt::format("{}/gel_{}.obj", objsDir, step);
            obj->save(pth1);
            gel->save<false>(pth2);
        }
    };
    return zs::make_tuple(solver, obj, gel, move, save_scene);
}

template <typename SolverPtrT, typename ObjPtrT, typename GelPtrT, typename MoveFuncT, typename SaveSceneFuncT>
void task(SolverPtrT solver, ObjPtrT obj, GelPtrT gel, MoveFuncT move, SaveSceneFuncT save_scene)
{
    using vec3 = tacipc::ABDSolver::vec3;
    using mat4 = tacipc::ABDSolver::mat4;

    using RigidTriBody = tacipc::RigidTriangleBody<T>;
    using RigidTetBody = tacipc::RigidTetBody<T>;
    using SoftTriBody = tacipc::SoftTriangleBody<T>;
    using SoftTetBody = tacipc::SoftTetBody<T>;
    using TriRigidBodyProperties = tacipc::RigidBodyProperties<T>;
    using TetRigidBodyProperties = tacipc::RigidBodyProperties<T>;

    auto pol = zs::cuda_exec().device(0);

    tacipc::SimViewer viewer(*solver, pol, {}, {});
    viewer.setFPS(60);
    viewer.setFPStep(1);

    int step = 0;
    save_scene(step++);

    mat4 objMotion = mat4::identity();

    viewer.setSimCallback([&](){
        fmt::print("Starting simulation step {}\n", step);
        if (!move(objMotion, step))
        {
            // viewer.setPlaySim(false); // do not do this,already have mutex lock
            viewer.playSim = false;
            viewer.stepSim = false;
        }
    });

    viewer.setUpdateCallback( // after each simulation step
        [&]()mutable
        {
            save_scene(step++);
        }
    );
    fmt::print("viewer ready\n");

    viewer.launch();
}

template <typename SolverPtrT, typename ObjPtrT, typename GelPtrT, typename MoveFuncT, typename SaveSceneFuncT>
void task_no_gui(SolverPtrT solver, ObjPtrT obj, GelPtrT gel, MoveFuncT move, SaveSceneFuncT save_scene)
{
    using vec3 = tacipc::ABDSolver::vec3;
    using mat4 = tacipc::ABDSolver::mat4;

    using RigidTriBody = tacipc::RigidTriangleBody<T>;
    using RigidTetBody = tacipc::RigidTetBody<T>;
    using SoftTriBody = tacipc::SoftTriangleBody<T>;
    using SoftTetBody = tacipc::SoftTetBody<T>;
    using TriRigidBodyProperties = tacipc::RigidBodyProperties<T>;
    using TetRigidBodyProperties = tacipc::RigidBodyProperties<T>;
    
    auto pol = zs::cuda_exec().device(0);

    zs::CppTimer timer;
    T timeElapsed = 0;
    int step = 0;
    mat4 objMotion = mat4::identity();

    save_scene(step++);
    while (move(objMotion, step))
    {
        timer.tick();
        solver->step(pol, true);
        timer.tock(fmt::format("Simulation step {}", step));
        timeElapsed += timer.elapsed();
        fmt::print("Total simulation time elapsed: {}ms\n", timeElapsed);
        save_scene(step++);
    }

    fmt::print("Simulation done. Time elapsed: {}ms\n", timeElapsed);
}

int main(int argc, char **argv)
{
    if (argc < 18)
    {
        fmt::print("Usage: {} <expName> <enableGui> <enableCGSolver> <cgRel> <PNCap> <enableInversionPrevention> <gelPth> <objPth> <isBCPth> <fricMu> <moveType> <pressSteps> <pressDepth> <pressVel> <taskSteps> <moveVel> <dt>\n", argv[0]);
        return 1;
    }

    auto curArg = argv + 1;
    
    auto stonum = [](const std::string &s)
        {
            if constexpr (std::is_same_v<T, float>)
                return std::stof(s);
            else if constexpr (std::is_same_v<T, double>)
                return std::stod(s);
        };  

    std::string expName = *(curArg++);
    bool enableGui;
    {
        std::string arg = *(curArg++);
        enableGui = arg == "true";
    }
    bool enableCGSolver;
    {
        std::string arg = *(curArg++);
        enableCGSolver = arg == "true";
    }
    T cgRel = stonum(*(curArg++));
    int PNCap = std::stoi(*(curArg++));
    bool enableInversionPrevention;
    {
        std::string arg = *(curArg++);
        enableInversionPrevention = arg == "true";
    }
    std::string gelPth = *(curArg++);
    std::string objPth = *(curArg++);
    std::string isBCPth = *(curArg++);
    T fricMu = stonum(*(curArg++));
    std::string moveType = *(curArg++);
    int pressSteps = std::stoi(*(curArg++));
    T pressDepth = stonum(*(curArg++));
    T pressVel = stonum(*(curArg++));
    int taskSteps = std::stoi(*(curArg++));
    T moveVel = stonum(*(curArg++));
    T dt = stonum(*(curArg++));
   
    fmt::print("args: [expName: {}, enableGui: {}, enableCGSolver: {}, cgRel: {}, PNCap: {}, enableInversionPrevention: {}, gelPth: {}, objPth: {}, isBCPth: {}, fricMu: {}, moveType: {}, pressSteps: {}, pressDepth: {}, pressVel: {}, taskSteps: {}, moveVel: {}, dt: {}]\n", expName, enableGui, enableCGSolver, cgRel, PNCap, enableInversionPrevention, gelPth, objPth, isBCPth, fricMu, moveType, pressSteps, pressDepth, pressVel, taskSteps, moveVel, dt);

    auto [solver, obj, gel, move, save_scene] = prepare(expName, enableCGSolver, cgRel, PNCap, enableInversionPrevention, gelPth, objPth, isBCPth, fricMu, moveType, pressSteps, pressDepth, pressVel, taskSteps, moveVel, dt);
    if (enableGui)
        task(solver, obj, gel, move, save_scene);
    else
        task_no_gui(solver, obj, gel, move, save_scene);
}




    
