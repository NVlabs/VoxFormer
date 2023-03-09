// pybind libraries
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// C/C++ includes
#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>

// #include <omp.h>

namespace py = pybind11;

namespace Eigen {
    typedef Matrix<bool, Dynamic, 1> VectorXb;
    typedef Matrix<signed char,Dynamic,1> VectorXsc;
};


/*
 * @brief returns all the voxels that are traversed by a ray going from start to end
 * @param start : continous world position where the ray starts
 * @param end   : continous world position where the ray end
 * @return vector of voxel ids hit by the ray in temporal order
 *
 * J. Amanatides, A. Woo. A Fast Voxel Traversal Algorithm for Ray Tracing. Eurographics '87
 *
 * Code adapted from: https://github.com/francisengelmann/fast_voxel_traversal
 *
 * Warning:
 *   This is not production-level code.
 */
inline bool _voxel_traversal(std::vector<Eigen::Vector3i> & visited_voxels,
                             const Eigen::Vector3d & ray_start,
                             const Eigen::Vector3d & ray_end,
                             const Eigen::Vector3i & grid_size,
                             const double voxel_size)
{
    // Compute normalized ray direction.
    Eigen::Vector3d ray = ray_end-ray_start;
    // ray.normalize();

    // This id of the first/current voxel hit by the ray.
    // Using floor (round down) is actually very important,
    // the implicit int-casting will round up for negative numbers.
    Eigen::Vector3i current_voxel(floor(ray_start[0]/voxel_size),
                                  floor(ray_start[1]/voxel_size),
                                  floor(ray_start[2]/0.2f));

    // create aliases for indices of the current voxel
    int &vx = current_voxel[0], &vy = current_voxel[1], &vz = current_voxel[2];

    // create maximum indices for each dimension (set the boundaries)
    const int vxsize = grid_size[0], vysize = grid_size[1], vzsize = grid_size[2];

    // The id of the last voxel hit by the ray.
    // TODO: what happens if the end point is on a border?
    Eigen::Vector3i last_voxel(floor(ray_end[0]/voxel_size),
                               floor(ray_end[1]/voxel_size),
                               floor(ray_end[2]/0.2f));

    // In which direction the voxel ids are incremented.
    int stepX = (ray[0] >= 0) ? 1:-1; // correct
    int stepY = (ray[1] >= 0) ? 1:-1; // correct
    int stepZ = (ray[2] >= 0) ? 1:-1; // correct

    // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
    double next_voxel_boundary_x = (vx+stepX)*voxel_size; // correct
    double next_voxel_boundary_y = (vy+stepY)*voxel_size; // correct
    double next_voxel_boundary_z = (vz+stepZ)*0.2f; // correct

    // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
    // the value of t at which the ray crosses the first vertical voxel boundary
    double tMaxX = (ray[0]!=0) ? (next_voxel_boundary_x - ray_start[0])/ray[0] : DBL_MAX; //
    double tMaxY = (ray[1]!=0) ? (next_voxel_boundary_y - ray_start[1])/ray[1] : DBL_MAX; //
    double tMaxZ = (ray[2]!=0) ? (next_voxel_boundary_z - ray_start[2])/ray[2] : DBL_MAX; //

    // tDeltaX, tDeltaY, tDeltaZ --
    // how far along the ray we must move for the horizontal component to equal the width of a voxel
    // the direction in which we traverse the grid
    // can only be FLT_MAX if we never go in that direction
    double tDeltaX = (ray[0]!=0) ? voxel_size/ray[0]*stepX : DBL_MAX;
    double tDeltaY = (ray[1]!=0) ? voxel_size/ray[1]*stepY : DBL_MAX;
    double tDeltaZ = (ray[2]!=0) ? 0.2f/ray[2]*stepZ : DBL_MAX;

    // Note: I am not sure why there is a need to do this, but I am keeping it for now
    // possibly explained by: https://github.com/francisengelmann/fast_voxel_traversal/issues/6
    Eigen::Vector3i diff(0,0,0);
    bool neg_ray=false;
    if (vx!=last_voxel[0] && ray[0]<0) { diff[0]--; neg_ray=true; }
    if (vy!=last_voxel[1] && ray[1]<0) { diff[1]--; neg_ray=true; }
    if (vz!=last_voxel[2] && ray[2]<0) { diff[2]--; neg_ray=true; }
    visited_voxels.push_back(current_voxel);
    if (neg_ray) {
        current_voxel+=diff;
        visited_voxels.push_back(current_voxel);
    }

    // ray casting loop
    bool truncated = false;
    while (current_voxel != last_voxel) {
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                vx += stepX;
                truncated = (vx < 0 || vx >= vxsize);
                tMaxX += tDeltaX;
            } else {
                vz += stepZ;
                truncated = (vz < 0 || vz >= vzsize);
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                vy += stepY;
                truncated = (vy < 0 || vy >= vysize);
                tMaxY += tDeltaY;
            } else {
                vz += stepZ;
                truncated = (vz < 0 || vz >= vzsize);
                tMaxZ += tDeltaZ;
            }
        }
        if (truncated) break;
        visited_voxels.push_back(current_voxel);
    }
    return truncated;
}


std::tuple<Eigen::VectorXsc, Eigen::VectorXb, Eigen::VectorXb> _compute_visibility_and_masks(
    const Eigen::MatrixXf & original_points,
    const Eigen::MatrixXf & sampled_points,
    const Eigen::MatrixXf & sensor_origins,
    const Eigen::VectorXf & time_stamps,
    const Eigen::VectorXf & pc_range,
    const double voxel_size,
    const bool culling_sampled,
    const bool culling_original)
{
    // py::gil_scoped_acquire acquire;

    //
    const double & pxmin = pc_range[0];
    const double & pymin = pc_range[1];
    const double & pzmin = pc_range[2];
    const double & pxmax = pc_range[3];
    const double & pymax = pc_range[4];
    const double & pzmax = pc_range[5];

    //
    const int vxsize = (pxmax - pxmin) / voxel_size;
    const int vysize = (pymax - pymin) / voxel_size;
    const int vzsize = (pzmax - pzmin) / voxel_size;
    const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);

    //
    const int T = time_stamps.size() - 1;
    const int G1 = vxsize;
    const int G2 = vysize * G1;
    const int G3 = vzsize * G2;
    const int G4 = T * G3;

    //
    Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);

    //
    const signed char OCCUPIED = 1;
    const signed char UNKNOWN = 0;
    const signed char FREE = -1;
    Eigen::VectorXsc visibility = Eigen::VectorXsc::Constant(G4, UNKNOWN);

    //
    Eigen::VectorXb original_visible_point_mask = Eigen::VectorXb::Zero(original_points.rows());
    Eigen::VectorXb sampled_visible_point_mask = Eigen::VectorXb::Zero(sampled_points.rows());

    // go through all sampled points and put them into different bins based on the timestamps
    std::vector<std::vector<int>> original_indices (T, std::vector<int>());
    for (int i = 0, last_t = -1; i < original_points.rows(); ++ i) {
        const double pt = original_points(i,3);
        if (last_t >= 0 && time_stamps[last_t] <= pt && pt < time_stamps[last_t+1]) {
            original_indices[last_t].push_back(i);
        } else {
            for (int t = 0; t < T; ++ t) {
                if (time_stamps[t] <= pt && pt < time_stamps[t+1]) {
                    original_indices[t].push_back(i);
                    last_t = t;
                    break;
                }
            }
        }
    }

    // go through all sampled points and put them into different bins based on the timestamps
    std::vector<std::vector<int>> sampled_indices (T, std::vector<int>());
    for (int i = 0, last_t = -1; i < sampled_points.rows(); ++ i) {
        const double pt = sampled_points(i,3);
        if (last_t >= 0 && time_stamps[last_t] <= pt && pt < time_stamps[last_t+1]) {
            sampled_indices[last_t].push_back(i);
        } else {
            for (int t = 0; t < T; ++ t) {
                if (time_stamps[t] <= pt && pt < time_stamps[t+1]) {
                    sampled_indices[t].push_back(i);
                    last_t = t;
                    break;
                }
            }
        }
    }

    // sometimes we happen to sample objects with a shorter history
    // when this happens, we replicate lidar points from the closest timestamp to make up the missing sweeps
    // we do not explicitly replicate the points, we replicate the indices to the big point matrix
    // this is a poor man's trick as opposed to building a dense point cloud model for every object
    // t = 0 is the current time stamp
    for (int t = 0, last_t = -1; t < T; ++ t) {
        if (sampled_indices[t].size() > 0) {
            last_t = t;
        } else if (last_t >= 0) {
            sampled_indices[t] = sampled_indices[last_t];
        }
    }

    // #pragma omp parallel for
    for (int t = 0; t < T; ++ t) {
        // set up update variable and sensor origin
        Eigen::VectorXf update = Eigen::VectorXf::Zero(G3);
        Eigen::Vector3d origin = (sensor_origins.row(t) - offset3d).cast<double>();

        // first, we compute a voxel mask for sampled points
        std::vector<bool> sampled_occupied_voxel_mask(G3, false);
        for (const int & i : sampled_indices[t]) {
            const int vx = floor((sampled_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((sampled_points(i,1) - pymin) / voxel_size);
            const int vz = floor((sampled_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int vidx = vz * G2 + vy * G1 + vx;
                sampled_occupied_voxel_mask[vidx] = true;
            }
        }

        // cast original rays and stop at voxels occupied by sampled points
        for (const int & i : original_indices[t]) {
            Eigen::Vector3d point = (original_points.row(i) - offset4d).head(3).cast<double>();
            std::vector<Eigen::Vector3i> visited_voxels;
            bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
            const int M = visited_voxels.size();
            for (int j = 0; j < M; ++ j) {
                const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
                const int vidx = vz * G2 + vy * G1 + vx;
                const int midx = t * G3 + vidx;
                if (sampled_occupied_voxel_mask[vidx]) {
                    if (culling_original) {
                        visibility[midx] = OCCUPIED;
                        break;
                    } else {
                        visibility[midx] = FREE;
                    }
                } else if (visibility[midx] == OCCUPIED) {
                    continue;
                } else {
                    visibility[midx] = (j==M-1 && !truncated) ? OCCUPIED : FREE ;
                }
            }
        }

        // second, we compute a voxel mask for original points
        std::vector<bool> original_occupied_voxel_mask(G3, false);
        for (const int & i : original_indices[t]) {
            const int vx = floor((original_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((original_points(i,1) - pymin) / voxel_size);
            const int vz = floor((original_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int vidx = vz * G2 + vy * G1 + vx;
                original_occupied_voxel_mask[vidx] = true;
            }
        }

        // cast sampled rays and mark voxels occupied by original points
        for (const int & i : sampled_indices[t]) {
            Eigen::Vector3d point = (sampled_points.row(i) - offset4d).head(3).cast<double>();
            std::vector<Eigen::Vector3i> visited_voxels;
            bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
            const int M = visited_voxels.size();
            for (int j = 0; j < M; ++ j) {
                const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
                const int vidx = vz * G2 + vy * G1 + vx;
                const int midx = t * G3 + vidx;
                if (original_occupied_voxel_mask[vidx]) {
                    if (culling_sampled) {
                        visibility[midx] = OCCUPIED;
                        break;
                    } else {
                        visibility[midx] = FREE;
                    }
                } else if (visibility[midx] == OCCUPIED) {
                    continue;
                } else {
                    visibility[midx] = (j==M-1 && !truncated) ? OCCUPIED : FREE ;
                }
            }
        }

        // go through the final visibility update and compute a binary mask for original points
        for (const int & i : original_indices[t]) {
            const int vx = floor((original_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((original_points(i,1) - pymin) / voxel_size);
            const int vz = floor((original_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int midx = t * G3 + vz * G2 + vy * G1 + vx;
                original_visible_point_mask[i] = (visibility[midx] == OCCUPIED);
            }
        }

        // go through the final visibility update and compute a binary mask for sampled points
        for (const int & i : sampled_indices[t]) {
            const int vx = floor((sampled_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((sampled_points(i,1) - pymin) / voxel_size);
            const int vz = floor((sampled_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int midx = t * G3 + vz * G2 + vy * G1 + vx;
                sampled_visible_point_mask[i] = (visibility[midx] == OCCUPIED);
            }
        }
    }
    return std::make_tuple(visibility, original_visible_point_mask, sampled_visible_point_mask);
}


Eigen::VectorXsc _compute_visibility(const Eigen::MatrixXf & original_points,
                                     const Eigen::MatrixXf & sensor_origins,
                                     const Eigen::VectorXf & time_stamps,
                                     const Eigen::VectorXf & pc_range,
                                     const double voxel_size)
{
    // py::gil_scoped_acquire acquire;

    //
    const double & pxmin = pc_range[0];
    const double & pymin = pc_range[1];
    const double & pzmin = pc_range[2];
    const double & pxmax = pc_range[3];
    const double & pymax = pc_range[4];
    const double & pzmax = pc_range[5];

    //
    const int vxsize = (pxmax - pxmin) / voxel_size;
    const int vysize = (pymax - pymin) / voxel_size;
    const int vzsize = (pzmax - pzmin) / voxel_size;
    const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);

    //
    const int T = time_stamps.size() - 1;
    const int G1 = vxsize;
    const int G2 = vysize * G1;
    const int G3 = vzsize * G2;
    const int G4 = T * G3;

    //
    Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);

    //
    const signed char OCCUPIED = 1;
    const signed char UNKNOWN = 0;
    const signed char FREE = -1;
    Eigen::VectorXsc visibility = Eigen::VectorXsc::Constant(G4, UNKNOWN);

    // go through all sampled points and put them into different bins based on the timestamps
    std::vector<std::vector<int>> original_indices (T, std::vector<int>());
    for (int i = 0, last_t = -1; i < original_points.rows(); ++ i) {
        const double pt = original_points(i,3);
        if (last_t >= 0 && time_stamps[last_t] <= pt && pt < time_stamps[last_t+1]) {
            original_indices[last_t].push_back(i);
        } else {
            for (int t = 0; t < T; ++ t) {
                if (time_stamps[t] <= pt && pt < time_stamps[t+1]) {
                    original_indices[t].push_back(i);
                    last_t = t;
                    break;
                }
            }
        }
    }

    // #pragma omp parallel for
    for (int t = 0; t < T; ++ t) {
        // COMPUTE VISIBILITY
        Eigen::VectorXf update = Eigen::VectorXf::Zero(G3);
        Eigen::Vector3d origin = (sensor_origins.row(t) - offset3d).cast<double>();

        for (const int & i : original_indices[t]) {
            Eigen::Vector3d point = (original_points.row(i) - offset4d).head(3).cast<double>();
            std::vector<Eigen::Vector3i> visited_voxels;
            bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
            const int M = visited_voxels.size();
            for (int j = 0; j < M; ++ j) {
                const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
                const int vidx = vz * G2 + vy * G1 + vx;
                const int midx = t * G3 + vidx;
                if (visibility[midx] == OCCUPIED) {
                    continue;
                } else {
                    visibility[midx] = (j==M-1 && !truncated) ? OCCUPIED : FREE ;
                }
            }
        }
    }
    return visibility;
}

std::tuple<Eigen::VectorXf, Eigen::VectorXb, Eigen::VectorXb> _compute_logodds_and_masks(
    const Eigen::MatrixXf & original_points,
    const Eigen::MatrixXf & sampled_points,
    const Eigen::MatrixXf & sensor_origins,
    const Eigen::VectorXf & time_stamps,
    const Eigen::VectorXf & pc_range,
    const double voxel_size,
    const double lo_occupied,
    const double lo_free,
    const bool culling_original,
    const bool culling_sampled)
{
    // py::gil_scoped_acquire acquire;

    //
    const double & pxmin = pc_range[0];
    const double & pymin = pc_range[1];
    const double & pzmin = pc_range[2];
    const double & pxmax = pc_range[3];
    const double & pymax = pc_range[4];
    const double & pzmax = pc_range[5];

    //
    const int vxsize = (pxmax - pxmin) / voxel_size;
    const int vysize = (pymax - pymin) / voxel_size;
    const int vzsize = (pzmax - pzmin) / voxel_size;
    const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);

    //
    const int T = time_stamps.size() - 1;
    const int G1 = vxsize;
    const int G2 = vysize * G1;
    const int G3 = vzsize * G2;

    //
    Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);

    //
    Eigen::VectorXf logodds = Eigen::VectorXf::Zero(G3);

    //
    Eigen::VectorXb original_visible_point_mask = Eigen::VectorXb::Zero(original_points.rows());
    Eigen::VectorXb sampled_visible_point_mask = Eigen::VectorXb::Zero(sampled_points.rows());

    // go through all sampled points and put them into different bins based on the timestamps
    std::vector<std::vector<int>> original_indices (T, std::vector<int>());
    for (int i = 0, last_t = -1; i < original_points.rows(); ++ i) {
        const double pt = original_points(i,3);
        if (last_t >= 0 && time_stamps[last_t] <= pt && pt < time_stamps[last_t+1]) {
            original_indices[last_t].push_back(i);
        } else {
            for (int t = 0; t < T; ++ t) {
                if (time_stamps[t] <= pt && pt < time_stamps[t+1]) {
                    original_indices[t].push_back(i);
                    last_t = t;
                    break;
                }
            }
        }
    }

    // go through all sampled points and put them into different bins based on the timestamps
    std::vector<std::vector<int>> sampled_indices (T, std::vector<int>());
    for (int i = 0, last_t = -1; i < sampled_points.rows(); ++ i) {
        const double pt = sampled_points(i,3);
        if (last_t >= 0 && time_stamps[last_t] <= pt && pt < time_stamps[last_t+1]) {
            sampled_indices[last_t].push_back(i);
        } else {
            for (int t = 0; t < T; ++ t) {
                if (time_stamps[t] <= pt && pt < time_stamps[t+1]) {
                    sampled_indices[t].push_back(i);
                    last_t = t;
                    break;
                }
            }
        }
    }

    // sometimes we happen to sample objects with a shorter history
    // when this happens, we replicate lidar points from the closest timestamp to make up the missing sweeps
    // we do not explicitly replicate the points, we replicate the indices to the big point matrix
    // this is a poor man's trick as opposed to building a dense point cloud model for every object
    // t = 0 is the current time stamp
    for (int t = 0, last_t = -1; t < T; ++ t) {
        if (sampled_indices[t].size() > 0) {
            last_t = t;
        } else if (last_t >= 0) {
            sampled_indices[t] = sampled_indices[last_t];
        }
    }

    // #pragma omp parallel for num_threads(3)
    for (int t = 0; t < T; ++ t) {
        // set up update variable and sensor origin
        Eigen::VectorXf update = Eigen::VectorXf::Zero(G3);
        Eigen::Vector3d origin = (sensor_origins.row(t) - offset3d).cast<double>();

        // first, we compute a voxel mask for sampled points
        std::vector<bool> sampled_occupied_voxel_mask(G3, false);
        for (const int & i : sampled_indices[t]) {
            const int vx = floor((sampled_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((sampled_points(i,1) - pymin) / voxel_size);
            const int vz = floor((sampled_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int vidx = vz * G2 + vy * G1 + vx;
                sampled_occupied_voxel_mask[vidx] = true;
            }
        }

        // cast original rays and stop at voxels occupied by sampled points
        for (const int & i : original_indices[t]) {
            Eigen::Vector3d point = (original_points.row(i) - offset4d).head(3).cast<double>();
            std::vector<Eigen::Vector3i> visited_voxels;
            bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
            const int M = visited_voxels.size();
            for (int j = 0; j < M; ++ j) {
                const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
                const int vidx = vz * G2 + vy * G1 + vx;
                if (sampled_occupied_voxel_mask[vidx]) {
                    if (culling_original) {
                        update[vidx] = lo_occupied;
                        break;
                    } else {
                        update[vidx] = lo_free;
                    }
                } else if (update[vidx] > 0) {
                    continue;
                } else {
                    update[vidx] = (j==M-1 && !truncated) ? lo_occupied : lo_free;
                }
            }
        }

        // second, we compute a voxel mask for original points
        std::vector<bool> original_occupied_voxel_mask(G3, false);
        for (const int & i : original_indices[t]) {
            const int vx = floor((original_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((original_points(i,1) - pymin) / voxel_size);
            const int vz = floor((original_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int vidx = vz * G2 + vy * G1 + vx;
                original_occupied_voxel_mask[vidx] = true;
            }
        }

        // cast sampled rays and mark voxels occupied by original points
        for (const int & i : sampled_indices[t]) {
            Eigen::Vector3d point = (sampled_points.row(i) - offset4d).head(3).cast<double>();
            std::vector<Eigen::Vector3i> visited_voxels;
            bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
            const int M = visited_voxels.size();
            for (int j = 0; j < M; ++ j) {
                const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
                const int vidx = vz * G2 + vy * G1 + vx;
                if (original_occupied_voxel_mask[vidx]) {
                    if (culling_sampled) {
                        update[vidx] = lo_occupied;
                        break;
                    } else {
                        update[vidx] = lo_free;
                    }
                } else if (update[vidx] > 0) {
                    continue;
                } else {
                    update[vidx] = (j==M-1 && !truncated) ? lo_occupied : lo_free;
                }
            }
        }

        // go through the final logodds update and compute a binary mask for original points
        for (const int & i : original_indices[t]) {
            const int vx = floor((original_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((original_points(i,1) - pymin) / voxel_size);
            const int vz = floor((original_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int vidx = vz * G2 + vy * G1 + vx;
                original_visible_point_mask[i] = (update[vidx] > 0);
            }
        }

        // go through the final logodds update and compute a binary mask for sampled points
        for (const int & i : sampled_indices[t]) {
            const int vx = floor((sampled_points(i,0) - pxmin) / voxel_size);
            const int vy = floor((sampled_points(i,1) - pymin) / voxel_size);
            const int vz = floor((sampled_points(i,2) - pzmin) / voxel_size);
            if (0 <= vz && vz < vzsize && 0 <= vy && vy < vysize && 0 <= vx && vx < vxsize) {
                const int vidx = vz * G2 + vy * G1 + vx;
                sampled_visible_point_mask[i] = (update[vidx] > 0);
            }
        }

        // update logodds
        // #pragma omp critical
        logodds += update;
    }

    return std::make_tuple(logodds, original_visible_point_mask, sampled_visible_point_mask);
}

Eigen::VectorXf _compute_logodds_dp(const Eigen::MatrixXf & original_points,
                                 const Eigen::MatrixXf & sensor_origins,
                                 const Eigen::VectorXf & pc_range,
                                 std::vector<int> original_indices,
                                 const double voxel_size,
                                 const double lo_occupied,
                                 const double lo_free)
{
    // py::gil_scoped_acquire acquire;

    //
    const double & pxmin = pc_range[0];
    const double & pymin = pc_range[1];
    const double & pzmin = pc_range[2];
    const double & pxmax = pc_range[3];
    const double & pymax = pc_range[4];
    const double & pzmax = pc_range[5];

    //
    const int vxsize = (pxmax - pxmin) / voxel_size;
    const int vysize = (pymax - pymin) / voxel_size;
    const int vzsize = ceil((pzmax - pzmin) / 0.2f);
    const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);

    //
    const int G1 = vxsize;
    const int G2 = vysize * G1;
    const int G3 = vzsize * G2;

    //
    Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);




    // COMPUTE VISIBILITY
    Eigen::VectorXf update = Eigen::VectorXf::Zero(G3);
    Eigen::Vector3d origin = (sensor_origins.row(0) - offset3d).cast<double>();

    for (const int & i : original_indices) {
        Eigen::Vector3d point = (original_points.row(i) - offset4d).head(3).cast<double>();
        //std::cout<<i<<" "<<point[0]<<" "<<point[1]<<" "<<point[2]<<" "<<std::endl;

        std::vector<Eigen::Vector3i> visited_voxels;
        bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
        const int M = visited_voxels.size();

        for (int j = 0; j < M; ++ j) {
            const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
            const int vidx = vz * G2 + vy * G1 + vx;
            if (update[vidx] > 0) { // if the voxel has been marked as occupied, move on
                continue;
            } else if (j == M-1 && !truncated) { // if it is the last voxel of an untrunc ray, mark it as occupied
                update[vidx] = lo_occupied;
            } else { // otherwise it must be a free voxel
                update[vidx] = lo_free;
            }
        }
    }

    return update;
}


Eigen::VectorXf _compute_logodds(const Eigen::MatrixXf & original_points,
                                 const Eigen::MatrixXf & sensor_origins,
                                 const Eigen::VectorXf & time_stamps,
                                 const Eigen::VectorXf & pc_range,
                                 const double voxel_size,
                                 const double lo_occupied,
                                 const double lo_free)
{
    // py::gil_scoped_acquire acquire;

    //
    const double & pxmin = pc_range[0];
    const double & pymin = pc_range[1];
    const double & pzmin = pc_range[2];
    const double & pxmax = pc_range[3];
    const double & pymax = pc_range[4];
    const double & pzmax = pc_range[5];

    //
    const int vxsize = (pxmax - pxmin) / voxel_size;
    const int vysize = (pymax - pymin) / voxel_size;
    const int vzsize = (pzmax - pzmin) / 0.2;
    const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);

    //
    const int T = time_stamps.size() - 1;
    const int G1 = vxsize;
    const int G2 = vysize * G1;
    const int G3 = vzsize * G2;

    //
    Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);

    //
    Eigen::VectorXf logodds = Eigen::VectorXf::Zero(G3);

    // go through all sampled points and put them into different bins based on the timestamps
    std::vector<std::vector<int>> original_indices (T, std::vector<int>());
    for (int i = 0, last_t = -1; i < original_points.rows(); ++ i) {
        const double pt = original_points(i,3);
        if (last_t >= 0 && time_stamps[last_t] <= pt && pt < time_stamps[last_t+1]) {
            original_indices[last_t].push_back(i);
        } else {
            for (int t = 0; t < T; ++ t) {
                if (time_stamps[t] <= pt && pt < time_stamps[t+1]) {
                    original_indices[t].push_back(i);
                    last_t = t;
                    break;
                }
            }
        }
    }

    // #pragma omp parallel for num_threads(3)
    for (int t = 0; t < T; ++ t) {
        // COMPUTE VISIBILITY
        Eigen::VectorXf update = Eigen::VectorXf::Zero(G3);
        Eigen::Vector3d origin = (sensor_origins.row(t) - offset3d).cast<double>();

        for (const int & i : original_indices[t]) {
            Eigen::Vector3d point = (original_points.row(i) - offset4d).head(3).cast<double>();
            std::vector<Eigen::Vector3i> visited_voxels;
            bool truncated = _voxel_traversal(visited_voxels, origin, point, grid_size, voxel_size);
            const int M = visited_voxels.size();
            for (int j = 0; j < M; ++ j) {
                const int &vx = visited_voxels[j][0], &vy = visited_voxels[j][1], &vz = visited_voxels[j][2];
                const int vidx = vz * G2 + vy * G1 + vx;
                if (update[vidx] > 0) { // if the voxel has been marked as occupied, move on
                    continue;
                } else if (j == M-1 && !truncated) { // if it is the last voxel of an untrunc ray, mark it as occupied
                    update[vidx] = lo_occupied;
                } else { // otherwise it must be a free voxel
                    update[vidx] = lo_free;
                }
            }
        }

        // #pragma omp critical
        logodds += update;
    }

    return logodds;
}


PYBIND11_MODULE(mapping, m) {
    m.doc() = "LiDAR voxel-based ray tracing";
    m.def("compute_visibility_and_masks",
          &_compute_visibility_and_masks,
          py::arg("original_points"),
          py::arg("sampled_points"),
          py::arg("sensor_origins"),
          py::arg("time_stamps"),
          py::arg("pc_range"),
          py::arg("voxel_size"),
          py::arg("culling_original")=true,
          py::arg("culling_sampled")=false
          );

    m.def("compute_visibility",
          &_compute_visibility,
          py::arg("original_points"),
          py::arg("sensor_origins"),
          py::arg("time_stamps"),
          py::arg("pc_range"),
          py::arg("voxel_size")
          );

    m.def("compute_logodds_and_masks",
          &_compute_logodds_and_masks,
          py::arg("original_points"),
          py::arg("sampled_points"),
          py::arg("sensor_origins"),
          py::arg("time_stamps"),
          py::arg("pc_range"),
          py::arg("voxel_size"),
          py::arg("lo_occupied")=std::log(0.7/(1-0.7)),
          py::arg("lo_free")=std::log(0.4/(1-0.4)),
          py::arg("culling_original")=true,
          py::arg("culling_sampled")=false
          );

    m.def("compute_logodds",
          &_compute_logodds,
          py::arg("original_points"),
          py::arg("sensor_origins"),
          py::arg("time_stamps"),
          py::arg("pc_range"),
          py::arg("voxel_size"),
          py::arg("lo_occupied")=std::log(0.7/(1-0.7)),
          py::arg("lo_free")=std::log(0.4/(1-0.4))
          );

    m.def("compute_logodds_dp",
          &_compute_logodds_dp,
          py::arg("original_points"),
          py::arg("sensor_origins"),
          py::arg("pc_range"),
          py::arg("indices"),
          py::arg("voxel_size"),
          py::arg("lo_occupied")=std::log(0.7/(1-0.7)),
          py::arg("lo_free")=std::log(0.4/(1-0.4))
          );
}
