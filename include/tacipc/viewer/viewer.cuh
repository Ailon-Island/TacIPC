#pragma once
#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <future>
#include <tacipc/types.hpp>
#include <tacipc/solver/Solver.cuh>

namespace tacipc
{

struct Frame
{
    using edge_t = std::tuple<Eigen::RowVector3d, Eigen::RowVector3d, Eigen::RowVector3d, unsigned>;
    using label_t = std::tuple<eigen::vec3d, std::string, unsigned>;

    Frame(const std::vector<const eigen::matX3d *> &verticesAll,
          const std::vector<const eigen::matX3i *> &facesAll,
          const std::vector<edge_t> &edges,
          const std::vector<label_t> &labels)
        : edges(edges), labels(labels) 
    {
        // allocate space
        {
            this->verticesAll.reserve(verticesAll.size());
            this->facesAll.reserve(facesAll.size());
        }
        for (const auto &vertices : verticesAll)
        {
            auto *v = new eigen::matX3d(*vertices);
            this->verticesAll.push_back(v);
        }
        for (const auto &faces : facesAll)
        {
            auto *f = new eigen::matX3i(*faces);
            this->facesAll.push_back(f);
        }
    }

    std::vector<const eigen::matX3d*> verticesAll;
    std::vector<const eigen::matX3i*> facesAll;
    std::vector<edge_t> edges;
    std::vector<label_t> labels;
};

class MeshViewer
{
    using edge_t = std::tuple<Eigen::RowVector3d, Eigen::RowVector3d, Eigen::RowVector3d, unsigned>;
    using label_t = std::tuple<eigen::vec3d, std::string, unsigned>;

  public:
    MeshViewer(const eigen::matX3d *vertices,
               const eigen::matX3i *faces);
    MeshViewer(const std::vector<const eigen::matX3d *> &verticesAll,
               const std::vector<const eigen::matX3i *> &facesAll);
    MeshViewer(std::vector<const eigen::matX3d *> &&verticesAll,
               std::vector<const eigen::matX3i *> &&facesAll);
    MeshViewer();

    void initialize();
    void launch();
    void shutdown()
    {
        glfwSetWindowShouldClose(iglViewer.window, 1);
    }

    void setCallback(std::function<void(void)> callback);
    void setMenuCallback(std::function<void(void)> callback);

    igl::opengl::glfw::Viewer &viewer();

    void setFPS(int fps);
    void setFontSize(int size) { labelSize = size; }

  protected:
    std::vector<Frame> frames;
    int curFrame = -1;
    std::vector<const eigen::matX3d *> verticesAll;
    std::vector<const eigen::matX3i *> facesAll;
    std::vector<int> ids;
    unsigned visualizeFlags = 0;
    std::vector<edge_t> edges;
    std::vector<label_t> labels;
    igl::opengl::glfw::Viewer iglViewer;
    igl::opengl::glfw::imgui::ImGuiPlugin imGuiPlugin;
    igl::opengl::glfw::imgui::ImGuiMenu imGuiMenu;
    std::function<void()> callback = [] {};
    std::function<void()> menuCallback = [] {};

    int labelSize = 6;
};

class ABDSolver;

class SimViewer : public MeshViewer
{
  public:
    SimViewer(ABDSolver &solver, ABDSolver::pol_t &pol, const std::vector<const eigen::matX3d *> &verticesAll={}, const std::vector<const eigen::matX3i *> &facesAll={});
    SimViewer(ABDSolver &solver, ABDSolver::pol_t &pol, std::vector<const eigen::matX3d *> &&verticesAll={}, std::vector<const eigen::matX3i *> &&facesAll={});
    void initialize(ABDSolver::pol_t &pol);
    void updateVertexLabel();
    void captureFrame() { frames.emplace_back(verticesAll, facesAll, edges, labels); }
    void setSimCallback(std::function<void(void)> callback);
    void setUpdateCallback(std::function<void(void)> callback) { updateCallback = callback; }
    void setFPStep(std::size_t fpstep) { FPStep = fpstep; stepCounter %= FPStep; }
    void setPlaySim(bool play); 
    void step(); 
    bool playSim = false; // sync with sim thread
    bool stepSim = false; // sync with sim thread

  protected:
    ABDSolver &solver;
    ABDSolver::pol_t &pol;
    zs::CppTimer simTimer;
    double timeElapsed = 0;
    std::size_t stepCounter = 0;
    std::size_t FPStep = 1;
    unsigned vertexLabelFlag = 1;
    std::function<void()> _step;
    std::function<void()> simCallback = [] {}; // do before each simulation step
    std::function<void()> updateCallback = []{}; // do after each simulation step
    std::mutex simMtx;
    std::future<void> simFuture;  
    bool toUpdate = false; // sync with sim thread
};

}; // namespace tacipc
