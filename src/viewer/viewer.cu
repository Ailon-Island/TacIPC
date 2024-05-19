#include <tacipc/viewer/viewer.cuh>

namespace tacipc
{
MeshViewer::MeshViewer(const std::vector<const eigen::matX3d *> &verticesAll,
                       const std::vector<const eigen::matX3i *> &facesAll)
    : verticesAll(verticesAll), facesAll(facesAll)
{
    initialize();
}

MeshViewer::MeshViewer(std::vector<const eigen::matX3d *> &&verticesAll,
                       std::vector<const eigen::matX3i *> &&facesAll)
    : verticesAll(verticesAll), facesAll(facesAll)
{
    initialize();
}

MeshViewer::MeshViewer(const eigen::matX3d *vertices,
                       const eigen::matX3i *faces)
    : MeshViewer(std::vector<const eigen::matX3d *>{vertices,},
                 std::vector<const eigen::matX3i *>{faces,})
{
    initialize();
}

MeshViewer::MeshViewer()
    : MeshViewer(std::vector<const eigen::matX3d *>{}, std::vector<const eigen::matX3i *>{})
{
    initialize();
}

void MeshViewer::initialize()
{
    iglViewer.core().animation_max_fps = 60;
    iglViewer.core().is_animating = true;

    // glfwSetWindowCloseCallback(iglViewer.window, 
    //                             [](GLFWwindow *window) -> void
    //                             {
    //                                 fcloseall();
    //                             });
}

void MeshViewer::launch()
{
    // set meshes
    auto vs = verticesAll.begin();
    auto fs = facesAll.begin();
    for (;vs != verticesAll.end() && fs != facesAll.end();
         ++vs, ++fs)
    {
        if ((*fs)->rows() > 0)
        {
            auto id = iglViewer.append_mesh();
            ids.emplace_back(id);
            iglViewer.data().set_mesh(**vs, **fs);
            if (false) // very cool(?) colored mesh
            {
                auto C = **vs;
                C = ((*vs)->rowwise() - (*vs)->colwise().minCoeff()).array().rowwise()/
                    (((*vs)->colwise().maxCoeff() - (*vs)->colwise().minCoeff()).array() + std::numeric_limits<double>::epsilon());
                iglViewer.data().set_colors(C);
            }
            fmt::print("mesh {} has {} vertices and {} faces\n", id, (*vs)->rows(), (*fs)->rows());
        }
        else
        {
            ids.emplace_back(-1);
            fmt::print("mesh has {} vertices and no faces. skip adding to viewer\n", (*vs)->rows());
        }
    }

    // set callback
    iglViewer.callback_pre_draw =
        [&](igl::opengl::glfw::Viewer &viewer) -> bool
        {
            callback();

            auto showFrame = curFrame;

            viewer.data().label_size = labelSize;

            auto &vs = (showFrame == -1) ? verticesAll : frames[showFrame].verticesAll;
            auto &fs = (showFrame == -1) ? facesAll : frames[showFrame].facesAll;
            for (int id = 0; id < ids.size(); ++id)
            {
                if (fs[id]->rows() > 0)
                    viewer.data(ids[id]).set_mesh(*vs[id], *fs[id]);
            }

            viewer.data().clear_edges();
            viewer.data().clear_labels();
            auto &es = (showFrame == -1) ? edges : frames[showFrame].edges;
            auto &ls = (showFrame == -1) ? labels : frames[showFrame].labels;
            for (auto &edge : es)
            {
                if (std::get<3>(edge) & visualizeFlags)
                    viewer.data().add_edges(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
            }
            for (auto &label : ls)
            {
                if (std::get<2>(label) & visualizeFlags)
                    viewer.data().add_label(std::get<0>(label), std::get<1>(label));
            }
            return false;
        };

    printf("setting imgui...\n");
    iglViewer.data().show_custom_labels = true;

    // set imgui
    iglViewer.plugins.push_back(&imGuiPlugin);
    imGuiPlugin.widgets.push_back(&imGuiMenu);
    imGuiMenu.callback_draw_viewer_menu = [&]()
    {
        // printf("calling menu callback...\n");
        // Draw parent menu content
        // imGuiMenu.draw_viewer_menu();
        if (ImGui::CollapsingHeader("IGL"))
        {
            imGuiMenu.draw_viewer_menu();
        }
        menuCallback();
        // printf("menu callback called\n");
    };

    printf("lauching viewer...\n");
    iglViewer.launch();
}

void MeshViewer::setCallback(std::function<void(void)> callback)
{
    this->callback = callback;
}

void MeshViewer::setMenuCallback(std::function<void(void)> callback)
{
    this->menuCallback = callback;
}

igl::opengl::glfw::Viewer &MeshViewer::viewer() { return iglViewer; }

void MeshViewer::setFPS(int fps) { iglViewer.core().animation_max_fps = fps; }

SimViewer::SimViewer(ABDSolver &solver, ABDSolver::pol_t &pol, const std::vector<const eigen::matX3d *> &verticesAll, const std::vector<const eigen::matX3i *> &facesAll)
    : MeshViewer(verticesAll, facesAll), solver(solver), pol(pol)
{
    initialize(pol);
}

SimViewer::SimViewer(ABDSolver &solver, ABDSolver::pol_t &pol, std::vector<const eigen::matX3d *> &&verticesAll, std::vector<const eigen::matX3i *> &&facesAll)
    : MeshViewer(verticesAll, facesAll), solver(solver), pol(pol)
{
    initialize(pol);
}

void SimViewer::setSimCallback(std::function<void(void)> callback)
{
    simCallback = callback;
}

void SimViewer::updateVertexLabel()
{
    labels.clear();
    // if (showVertexLabel)
    {
        for (auto &handle : solver.rigidBodies)
        {
            auto &verts = handle.meshVerts();
            auto svs = handle.svInds().clone({zs::memsrc_e::host, -1});
            for (int i = 0; i < svs.size(); ++i)
            {
                int vi = svs[i] + handle.voffset();
                labels.emplace_back(verts.row(i), std::to_string(vi), vertexLabelFlag);
            }
        }
        for (auto &handle : solver.softBodies)
        {
            auto &verts = handle.meshVerts();
            auto svs = handle.svInds().clone({zs::memsrc_e::host, -1});
            for (int i = 0; i < svs.size(); ++i)
            {
                int vi = svs[i] + handle.voffset();
                labels.emplace_back(verts.row(i), std::to_string(vi), vertexLabelFlag);
            }
        }
    }
}

void SimViewer::initialize(ABDSolver::pol_t &pol)
{
    MeshViewer::initialize();

    for (auto &handle : solver.rigidBodies)
    {
        this->verticesAll.emplace_back(&handle.meshVerts());
        this->facesAll.emplace_back(&handle.meshTris());
    }
    for (auto &handle : solver.softBodies)
    {
        this->verticesAll.emplace_back(&handle.meshVerts());
        this->facesAll.emplace_back(&handle.meshTris());
    }
    updateVertexLabel();
    captureFrame();

    setCallback([&]() {
        ++stepCounter;
        stepCounter %= FPStep;

        auto lock = std::unique_lock<std::mutex>(simMtx, std::try_to_lock);
        if (lock.owns_lock()) // if simulation is not stepping
        {
            zs::prepare_context(zs::mem_device, 0);
            // update data
            if (toUpdate)
            {
                // solver.update(pol);
                updateCallback();
                updateVertexLabel();
                captureFrame();

                toUpdate = false;
            }
            lock.unlock();

            // step simulation
            if (stepSim)
            {
                if (stepCounter == 0)
                {
                    simFuture = std::async(std::launch::async, _step);
                }
            }
        }
        else // simulation is running, defer data update and simulation step
        {
            if (stepCounter == 0)
            {
                --stepCounter;
            }
        }
    });
    setMenuCallback([&](){
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (!playSim)
            {
                if (ImGui::Button("Play Simulation", ImVec2(-1, 0)))
                {
                    setPlaySim(true);
                }
            }
            else
            {
                if (ImGui::Button("Pause Simulation", ImVec2(-1, 0)))
                {
                    setPlaySim(false);
                }
            }
            if (ImGui::Button("Step Simulation", ImVec2(-1, 0)))
            {
                step();
            }
            ImGui::InputScalar("Frames per Step", ImGuiDataType_U32, &FPStep);
            bool showLastFrame = curFrame == -1;
            bool showLastFrameOld = showLastFrame;
            ImGui::Checkbox("Show Latest Frame", &showLastFrame);
            if (showLastFrame)
            {
                curFrame = -1;
            }
            else 
            {
                if (showLastFrameOld)
                    curFrame = frames.size() - 1;
                ImGui::SliderInt("Current Frame", &curFrame, 0, frames.size() - 1);
                const float button_size = ImGui::GetFrameHeight();
                ImGui::SameLine();
                if (ImGui::Button("-", ImVec2(button_size, button_size)))
                {
                    if (curFrame > 0)
                        --curFrame;
                }
                ImGui::SameLine();
                if (ImGui::Button("+", ImVec2(button_size, button_size)))
                {
                    if (curFrame < frames.size() - 1)
                        ++curFrame;
                }
            }
            if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::InputInt("Label Font Size", &labelSize, 1, 5);
                ImGui::CheckboxFlags("Show Vertex Label", &visualizeFlags, vertexLabelFlag);
                if (ImGui::CollapsingHeader("Mesh Visibility"))
                {
                    for (int i = 0; i < ids.size(); ++i)
                    {
                        if (ids[i] != -1)
                            ImGui::CheckboxFlags(fmt::format("Mesh {}", i).c_str(), &iglViewer.data_list[ids[i]].is_visible, iglViewer.core_list[0].id);
                    }
                }
            }
        }
    });

    _step = [&]() {
                auto lock = std::unique_lock<std::mutex>(simMtx);
                if (toUpdate) // update first
                    return;
                if (!stepSim)
                    return;
                zs::prepare_context(zs::mem_device, 0);
                simCallback();
                stepSim = playSim;   
                simTimer.tick();
                solver.step(pol, true);
                simTimer.tock("current simulation step");
                timeElapsed += simTimer.elapsed();
                fmt::print("Total simulation time for {} frames: {}ms\n", frames.size(), timeElapsed);
                toUpdate = true;
            };
}
void SimViewer::setPlaySim(bool play)
{
    auto lock = std::unique_lock<std::mutex>(simMtx);
    playSim = play;
    stepSim = play;
}
void SimViewer::step()
{
    auto lock = std::unique_lock<std::mutex>(simMtx, std::try_to_lock);
    if (lock.owns_lock())
    {
        if (toUpdate) // update first
            return;
        if (playSim)
            return;
        stepSim = true;
        stepCounter = 0;
        lock.unlock();
        // fmt::print("launch step\n");
        // simFuture = std::async(std::launch::async, _step);
    }
}
} // namespace tacipc