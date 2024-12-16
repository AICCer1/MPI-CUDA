#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <mpi.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double pi = M_PI; 

double compute_a_t(double Lx, double Ly, double Lz) {
    return pi * sqrt(1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
}

struct SimulationParams {
    int N;         
    double tau;    
    int K;         
    double T;     
    double Lx, Ly, Lz; 
    double hx, hy, hz;  
    double a_t;     

    void initialize(int n, double t, int k) {
        N = n;
        tau = t;
        K = k;
        T = tau * K;
        Lx = Ly = Lz = 1.0;
        hx = hy = hz = Lx / N;
        a_t = compute_a_t(Lx, Ly, Lz);
    }
};

struct DirInfo {
    int send_idx;      
    int recv_idx;      
    int stride;        
    int size1, size2;  
    int tag_send;      
    int tag_recv;      
};

int compute_local_size(int global_size, int num_procs, int proc_coord) {
    int base_size = global_size / num_procs;
    int remainder = global_size % num_procs;
    return base_size + (proc_coord < remainder ? 1 : 0);
}

int compute_start_index(int global_size, int num_procs, int proc_coord) {
    int base_size = global_size / num_procs;
    int remainder = global_size % num_procs;
    return proc_coord * base_size + min(proc_coord, remainder);
}

struct ProcessGrid {
    int dims[3];        
    int coords[3];    
    int local_nx;      
    int local_ny;      
    int local_nz;     
    int start_x;      
    int start_y;      
    int start_z;     
    bool is_x_left_boundary;
    bool is_x_right_boundary;
    bool is_y_bottom_boundary;
    bool is_y_top_boundary;
    bool is_z_front_boundary;
    bool is_z_back_boundary;
    int neighbors[6];  
    vector<DirInfo> dir_infos;

    void initialize(const SimulationParams& params, const int coords[3], MPI_Comm cart_comm) {
        this->coords[0] = coords[0];
        this->coords[1] = coords[1];
        this->coords[2] = coords[2];

        local_nx = compute_local_size(params.N - 1, dims[0], coords[0]);
        local_ny = compute_local_size(params.N - 1, dims[1], coords[1]);
        local_nz = compute_local_size(params.N - 1, dims[2], coords[2]);

        start_x = compute_start_index(params.N - 1, dims[0], coords[0]);
        start_y = compute_start_index(params.N - 1, dims[1], coords[1]);
        start_z = compute_start_index(params.N - 1, dims[2], coords[2]);

        MPI_Cart_shift(cart_comm, 0, 1, &neighbors[1], &neighbors[0]); 
        MPI_Cart_shift(cart_comm, 1, 1, &neighbors[3], &neighbors[2]); 
        MPI_Cart_shift(cart_comm, 2, 1, &neighbors[5], &neighbors[4]); 
    
        dir_infos = vector<DirInfo>{
            // X
            {local_nx, local_nx + 1, local_nz + 2, local_ny + 2, local_nz + 2, 0, 1},
            {1, 0, local_nz + 2, local_ny + 2, local_nz + 2, 1, 0},
            // Y
            {local_ny, local_ny + 1, local_nz + 2, local_nx + 2, local_nz + 2, 2, 3},
            {1, 0, local_nz + 2, local_nx + 2, local_nz + 2, 3, 2},
            // Z
            {local_nz, local_nz + 1, local_ny + 2, local_nx + 2, local_ny + 2, 4, 5},
            {1, 0, local_ny + 2, local_nx + 2, local_ny + 2, 5, 4}
        };
    }

    void set_boundaries(int N) {
        is_x_left_boundary = (start_x == 0);
        is_x_right_boundary = (start_x + local_nx == N - 1);
        is_y_bottom_boundary = (start_y == 0);
        is_y_top_boundary = (start_y + local_ny == N - 1);
        is_z_front_boundary = (start_z == 0);
        is_z_back_boundary = (start_z + local_nz == N - 1);
    }
};

inline int compute_index(int i, int j, int k, int local_ny, int local_nz) {
    return i * (local_ny + 2) * (local_nz + 2) + j * (local_nz + 2) + k;
}

struct Error {
    double max_error = 0.0;
    double l2_error = 0.0;
    size_t count = 0;

    void clear() {
        max_error = l2_error = 0.0;
        count = 0;
    }

    void update(double err) {
        double e = fabs(err);
        max_error = std::max(max_error, e);
        l2_error += e * e;
        count++;
    }

    void reduce(MPI_Comm comm) {
        double global_max_error;
        double global_l2_error;
        size_t global_count;

        MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&l2_error, &global_l2_error, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

        max_error = global_max_error;
        l2_error = global_l2_error;
        count = global_count;
    }

    double get_max_error() const { return max_error; }
    double get_l2_error() const { return sqrt(l2_error / count); }
};

double compute_u_analytical(double x, double y, double z, double t, const SimulationParams& params) {
    return sin(pi * x / params.Lx) *
           sin(pi * y / params.Ly) *
           sin(pi * z / params.Lz) *
           cos(params.a_t * t);
}

void start_boundary_exchange(vector<double>& u, const ProcessGrid& grid, MPI_Comm cart_comm,
                           vector<vector<double>>& send_buffers,
                           vector<vector<double>>& recv_buffers,
                           vector<MPI_Request>& requests) {
    
    requests.clear();  
    for (int dir = 0; dir < 6; ++dir) {
        if (grid.neighbors[dir] != MPI_PROC_NULL) {
            requests.push_back(MPI_REQUEST_NULL); 
            requests.push_back(MPI_REQUEST_NULL);  
        }
    }

    for (int dir = 0; dir < 6; ++dir) {
        if (grid.neighbors[dir] != MPI_PROC_NULL) {
            const auto& info = grid.dir_infos[dir];
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < info.size1; ++i) {
                for (int j = 0; j < info.size2; ++j) {
                    int buffer_idx = i * info.stride + j;
                    int u_idx;

                    if (dir < 2) { 
                        u_idx = compute_index(info.send_idx, i, j, grid.local_ny, grid.local_nz);
                    } else if (dir < 4) { 
                        u_idx = compute_index(i, info.send_idx, j, grid.local_ny, grid.local_nz);
                    } else { 
                        u_idx = compute_index(i, j, info.send_idx, grid.local_ny, grid.local_nz);
                    }

                    send_buffers[dir][buffer_idx] = u[u_idx];
                }
            }
        }
    }

    int request_count = 0;
    for (int dir = 0; dir < 6; ++dir) {
        if (grid.neighbors[dir] != MPI_PROC_NULL) {
            const auto& info = grid.dir_infos[dir];
            
            MPI_Isend(send_buffers[dir].data(), (int)send_buffers[dir].size(), MPI_DOUBLE,
                     grid.neighbors[dir], info.tag_send, cart_comm, &requests[request_count++]);
            MPI_Irecv(recv_buffers[dir].data(), (int)recv_buffers[dir].size(), MPI_DOUBLE,
                     grid.neighbors[dir], info.tag_recv, cart_comm, &requests[request_count++]);
        }
    }
}

void finish_boundary_exchange(vector<double>& u, const ProcessGrid& grid,
                            vector<vector<double>>& recv_buffers,
                            vector<MPI_Request>& requests) {
    if (!requests.empty()) {
        vector<MPI_Status> statuses(requests.size());
        MPI_Waitall((int)requests.size(), requests.data(), statuses.data());
    }

    for (int dir = 0; dir < 6; ++dir) {
        if (grid.neighbors[dir] != MPI_PROC_NULL) {
            const auto& info = grid.dir_infos[dir];
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < info.size1; ++i) {
                for (int j = 0; j < info.size2; ++j) {
                    int buffer_idx = i * info.stride + j;
                    int u_idx;

                    if (dir < 2) {
                        u_idx = compute_index(info.recv_idx, i, j, grid.local_ny, grid.local_nz);
                    } else if (dir < 4) {
                        u_idx = compute_index(i, info.recv_idx, j, grid.local_ny, grid.local_nz);
                    } else {
                        u_idx = compute_index(i, j, info.recv_idx, grid.local_ny, grid.local_nz);
                    }

                    u[u_idx] = recv_buffers[dir][buffer_idx];
                }
            }
        }
    }
    
    requests.clear();
}



void apply_boundary_conditions(vector<double>& u, const ProcessGrid& grid) {
    
    if (grid.is_x_left_boundary) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j <= grid.local_ny + 1; ++j) {
            for (int k = 0; k <= grid.local_nz + 1; ++k) {
                u[compute_index(0, j, k, grid.local_ny, grid.local_nz)] = 0.0;
            }
        }
    }
    if (grid.is_x_right_boundary) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j <= grid.local_ny + 1; ++j) {
            for (int k = 0; k <= grid.local_nz + 1; ++k) {
                u[compute_index(grid.local_nx + 1, j, k, grid.local_ny, grid.local_nz)] = 0.0;
            }
        }
    }

    if (grid.is_y_bottom_boundary) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= grid.local_nx + 1; ++i) {
            for (int k = 0; k <= grid.local_nz + 1; ++k) {
                u[compute_index(i, 0, k, grid.local_ny, grid.local_nz)] = 0.0;
            }
        }
    }
    if (grid.is_y_top_boundary) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= grid.local_nx + 1; ++i) {
            for (int k = 0; k <= grid.local_nz + 1; ++k) {
                u[compute_index(i, grid.local_ny + 1, k, grid.local_ny, grid.local_nz)] = 0.0;
            }
        }
    }

    if (grid.is_z_front_boundary) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= grid.local_nx + 1; ++i) {
            for (int j = 0; j <= grid.local_ny + 1; ++j) {
                u[compute_index(i, j, 0, grid.local_ny, grid.local_nz)] = 0.0;
            }
        }
    }
    if (grid.is_z_back_boundary) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= grid.local_nx + 1; ++i) {
            for (int j = 0; j <= grid.local_ny + 1; ++j) {
                u[compute_index(i, j, grid.local_nz + 1, grid.local_ny, grid.local_nz)] = 0.0;
            }
        }
    }
}

void initialize_solution(vector<double>& u, const ProcessGrid& grid,
                         const SimulationParams& params, double t) {
    #pragma omp parallel for collapse(3)                        
    for (int i = 1; i <= grid.local_nx; ++i) {
        for (int j = 1; j <= grid.local_ny; ++j) {
            for (int k = 1; k <= grid.local_nz; ++k) {
                double x = (grid.start_x + i) * params.hx;
                double y = (grid.start_y + j) * params.hy;
                double z = (grid.start_z + k) * params.hz;

                int idx = compute_index(i, j, k, grid.local_ny, grid.local_nz);
                u[idx] = compute_u_analytical(x, y, z, t, params);
            }
        }
    }
}

void compute_first_timestep(vector<double>& u_next, const vector<double>& u_curr,
                            const ProcessGrid& grid, const SimulationParams& params) {
    #pragma omp parallel for collapse(3)                            
    for (int i = 1; i <= grid.local_nx; ++i) {
        for (int j = 1; j <= grid.local_ny; ++j) {
            for (int k = 1; k <= grid.local_nz; ++k) {
                int idx = compute_index(i, j, k, grid.local_ny, grid.local_nz);
                
                double laplacian =
                    (u_curr[compute_index(i - 1, j, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
                     u_curr[compute_index(i + 1, j, k, grid.local_ny, grid.local_nz)]) / (params.hx * params.hx) +
                    (u_curr[compute_index(i, j - 1, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
                     u_curr[compute_index(i, j + 1, k, grid.local_ny, grid.local_nz)]) / (params.hy * params.hy) +
                    (u_curr[compute_index(i, j, k - 1, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
                     u_curr[compute_index(i, j, k + 1, grid.local_ny, grid.local_nz)]) / (params.hz * params.hz);

                u_next[idx] = u_curr[idx] + 0.5 * params.tau * params.tau * laplacian;
            }
        }
    }
}

Error compute_error(const vector<double>& u_numerical, const ProcessGrid& grid,
                    const SimulationParams& params, double t) {
    Error error;
    for (int i = 1; i <= grid.local_nx; ++i) {
        for (int j = 1; j <= grid.local_ny; ++j) {
            for (int k = 1; k <= grid.local_nz; ++k) {
                double x = (grid.start_x + i) * params.hx;
                double y = (grid.start_y + j) * params.hy;
                double z = (grid.start_z + k) * params.hz;

                int idx = compute_index(i, j, k, grid.local_ny, grid.local_nz);
                double u_exact = compute_u_analytical(x, y, z, t, params);
                error.update(u_numerical[idx] - u_exact);
            }
        }
    }
    return error;
}


void prepare_boundary_data(const vector<double>& u, const ProcessGrid& grid,
                         vector<vector<double>>& send_buffers) {
    
    #pragma omp parallel for default(none) shared(u, grid, send_buffers)
    for (int dir = 0; dir < 6; ++dir) {
        const auto& info = grid.dir_infos[dir];
        
        for (int i = 0; i < info.size1; ++i) {
            for (int j = 0; j < info.size2; ++j) {
                int buffer_idx = i * info.stride + j;
                int u_idx;

                if (dir < 2) { // X direction
                    u_idx = compute_index(info.send_idx, i, j, grid.local_ny, grid.local_nz);
                } else if (dir < 4) { // Y direction
                    u_idx = compute_index(i, info.send_idx, j, grid.local_ny, grid.local_nz);
                } else { // Z direction
                    u_idx = compute_index(i, j, info.send_idx, grid.local_ny, grid.local_nz);
                }

                send_buffers[dir][buffer_idx] = u[u_idx];
            }
        }
    }
}

void start_communication(const ProcessGrid& grid, MPI_Comm cart_comm,
                        vector<vector<double>>& send_buffers,
                        vector<vector<double>>& recv_buffers,
                        vector<MPI_Request>& requests) {
    requests.clear();
    for (int dir = 0; dir < 6; ++dir) {
        if (grid.neighbors[dir] != MPI_PROC_NULL) {
            requests.push_back(MPI_REQUEST_NULL);
            requests.push_back(MPI_REQUEST_NULL);
        }
    }

    int request_count = 0;
    for (int dir = 0; dir < 6; ++dir) {
        if (grid.neighbors[dir] != MPI_PROC_NULL) {
            const auto& info = grid.dir_infos[dir];
            
            MPI_Isend(send_buffers[dir].data(), (int)send_buffers[dir].size(), MPI_DOUBLE,
                     grid.neighbors[dir], info.tag_send, cart_comm, &requests[request_count++]);
            MPI_Irecv(recv_buffers[dir].data(), (int)recv_buffers[dir].size(), MPI_DOUBLE,
                     grid.neighbors[dir], info.tag_recv, cart_comm, &requests[request_count++]);
        }
    }
}

void compute_timestep_deep_interior(vector<double>& u_next, const vector<double>& u_curr,
                                  const vector<double>& u_prev, const ProcessGrid& grid,
                                  const SimulationParams& params) {
    const double hx2 = params.hx * params.hx;
    const double hy2 = params.hy * params.hy;
    const double hz2 = params.hz * params.hz;
    const double tau2 = params.tau * params.tau;
    
    #pragma omp parallel for collapse(3)
    
    for (int i = 2; i <= grid.local_nx - 1; ++i) {
        for (int j = 2; j <= grid.local_ny - 1; ++j) {
            for (int k = 2; k <= grid.local_nz - 1; ++k) {
                int idx = compute_index(i, j, k, grid.local_ny, grid.local_nz);
                
                double laplacian =
                    (u_curr[compute_index(i - 1, j, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
                     u_curr[compute_index(i + 1, j, k, grid.local_ny, grid.local_nz)]) / hx2 +
                    (u_curr[compute_index(i, j - 1, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
                     u_curr[compute_index(i, j + 1, k, grid.local_ny, grid.local_nz)]) / hy2 +
                    (u_curr[compute_index(i, j, k - 1, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
                     u_curr[compute_index(i, j, k + 1, grid.local_ny, grid.local_nz)]) / hz2;

                u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + tau2 * laplacian;
            }
        }
    }
}

void compute_timestep_near_boundary(vector<double>& u_next, const vector<double>& u_curr,
                                  const vector<double>& u_prev, const ProcessGrid& grid,
                                  const SimulationParams& params) {
    const double hx2 = params.hx * params.hx;
    const double hy2 = params.hy * params.hy;
    const double hz2 = params.hz * params.hz;
    const double tau2 = params.tau * params.tau;

    auto compute_point = [&](int i, int j, int k) {
        int idx = compute_index(i, j, k, grid.local_ny, grid.local_nz);
        double laplacian =
            (u_curr[compute_index(i - 1, j, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
             u_curr[compute_index(i + 1, j, k, grid.local_ny, grid.local_nz)]) / hx2 +
            (u_curr[compute_index(i, j - 1, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
             u_curr[compute_index(i, j + 1, k, grid.local_ny, grid.local_nz)]) / hy2 +
            (u_curr[compute_index(i, j, k - 1, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] +
             u_curr[compute_index(i, j, k + 1, grid.local_ny, grid.local_nz)]) / hz2;
        u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + tau2 * laplacian;
    };

    // X
    #pragma omp parallel for collapse(2)
    for (int j = 1; j <= grid.local_ny; ++j) {
        for (int k = 1; k <= grid.local_nz; ++k) {
            compute_point(1, j, k);
            compute_point(grid.local_nx, j, k);
        }
    }

    // Y
    #pragma omp parallel for collapse(2)
    for (int i = 2; i < grid.local_nx; ++i) {
        for (int k = 1; k <= grid.local_nz; ++k) {
            compute_point(i, 1, k);
            compute_point(i, grid.local_ny, k);
        }
    }

    // Z
    #pragma omp parallel for collapse(2)
    for (int i = 2; i < grid.local_nx; ++i) {
        for (int j = 2; j < grid.local_ny; ++j) {
            compute_point(i, j, 1);
            compute_point(i, j, grid.local_nz);
        }
    }
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
       
        double init_time = 0.0;
        double computation_time = 0.0;
        double communication_time = 0.0;
        double boundary_time = 0.0;
        
       
        double total_start_time = MPI_Wtime();
        double start_time = MPI_Wtime();

        if (argc < 5) {
            if (rank == 0) {
                cout << "Usage: " << argv[0] << " N tau K number_threads\n";
            }
            MPI_Finalize();
            return 1;
        }
        omp_set_num_threads(stoi(argv[4]));

        SimulationParams params;
        params.initialize(stoi(argv[1]), stod(argv[2]), stoi(argv[3]));

        ProcessGrid grid;
        grid.dims[0] = grid.dims[1] = grid.dims[2] = 0;
        MPI_Dims_create(size, 3, grid.dims);

        int periods[3] = {0, 0, 0};
        MPI_Comm cart_comm;
        MPI_Cart_create(MPI_COMM_WORLD, 3, grid.dims, periods, 0, &cart_comm);
        MPI_Cart_coords(cart_comm, rank, 3, grid.coords);

        grid.initialize(params, grid.coords, cart_comm);
        grid.set_boundaries(params.N);

        if (rank == 0) {
            cout << "Configuration:\n"
                 << "Grid size: " << params.N << "x" << params.N << "x" << params.N << "\n"
                 << "Process grid: " << grid.dims[0] << "x" << grid.dims[1] << "x" << grid.dims[2] << "\n"
                 << "Local grid size: " << grid.local_nx << "x" << grid.local_ny << "x" << grid.local_nz << "\n"
                 << "Time steps: " << params.K << "\n"
                 << "Threads per process: " << argv[4] << "\n";
        }

        
        size_t local_size = (grid.local_nx + 2) * (grid.local_ny + 2) * (grid.local_nz + 2);
        vector<double> u_prev(local_size, 0.0);
        vector<double> u_curr(local_size, 0.0);
        vector<double> u_next(local_size, 0.0);

        vector<vector<double>> send_buffers(6);
        vector<vector<double>> recv_buffers(6);

        // X
        send_buffers[0].resize((grid.local_ny + 2) * (grid.local_nz + 2)); 
        send_buffers[1].resize((grid.local_ny + 2) * (grid.local_nz + 2));
        recv_buffers[0].resize((grid.local_ny + 2) * (grid.local_nz + 2));
        recv_buffers[1].resize((grid.local_ny + 2) * (grid.local_nz + 2));

        // Y
        send_buffers[2].resize((grid.local_nx + 2) * (grid.local_nz + 2));
        send_buffers[3].resize((grid.local_nx + 2) * (grid.local_nz + 2));
        recv_buffers[2].resize((grid.local_nx + 2) * (grid.local_nz + 2));
        recv_buffers[3].resize((grid.local_nx + 2) * (grid.local_nz + 2));

        // Z
        send_buffers[4].resize((grid.local_nx + 2) * (grid.local_ny + 2));
        send_buffers[5].resize((grid.local_nx + 2) * (grid.local_ny + 2));
        recv_buffers[4].resize((grid.local_nx + 2) * (grid.local_ny + 2));
        recv_buffers[5].resize((grid.local_nx + 2) * (grid.local_ny + 2));

        vector<MPI_Request> requests;
        requests.reserve(12);

        init_time = MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

       
        initialize_solution(u_prev, grid, params, 0.0);
        computation_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        apply_boundary_conditions(u_prev, grid);
        boundary_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();
       
        start_boundary_exchange(u_prev, grid, cart_comm, send_buffers, recv_buffers, requests);
        finish_boundary_exchange(u_prev, grid, recv_buffers, requests);
        communication_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        compute_first_timestep(u_curr, u_prev, grid, params);
        computation_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        apply_boundary_conditions(u_curr, grid);
        boundary_time += MPI_Wtime() - start_time;

        MPI_Barrier(cart_comm);

        for (int n = 2; n <= params.K; ++n) {
            double t = n * params.tau;
          
           
            start_time = MPI_Wtime();
            prepare_boundary_data(u_curr, grid, send_buffers);
            communication_time += MPI_Wtime() - start_time;
            
           
            start_time = MPI_Wtime();
            start_communication(grid, cart_comm, send_buffers, recv_buffers, requests);
            communication_time += MPI_Wtime() - start_time;

            
            start_time = MPI_Wtime();
            compute_timestep_deep_interior(u_next, u_curr, u_prev, grid, params);
            computation_time += MPI_Wtime() - start_time;
            
           
            start_time = MPI_Wtime();
            finish_boundary_exchange(u_curr, grid, recv_buffers, requests);
            communication_time += MPI_Wtime() - start_time;
            
          
            start_time = MPI_Wtime();
            compute_timestep_near_boundary(u_next, u_curr, u_prev, grid, params);
            computation_time += MPI_Wtime() - start_time;

           
            start_time = MPI_Wtime();
            apply_boundary_conditions(u_next, grid);
            boundary_time += MPI_Wtime() - start_time;

            // if (n % (params.K / 10) == 0 || n == params.K) {
            //     start_time = MPI_Wtime();
            //     Error error = compute_error(u_next, grid, params, t);
            //     error.reduce(cart_comm);
            //     computation_time += MPI_Wtime() - start_time;
                
            //     if (rank == 0) {
            //         cout << "Step " << n
            //              << ", t = " << fixed << setprecision(6) << t
            //              << ", Max Error = " << scientific << setprecision(6) << error.get_max_error()
            //              << ", L2 Error = " << error.get_l2_error() << endl;
            //     }
            // }

            swap(u_prev, u_curr);
            swap(u_curr, u_next);
        }

        double total_time = MPI_Wtime() - total_start_time;

       
        double max_total_time, max_init_time, max_computation_time, 
               max_communication_time, max_boundary_time;
        
        MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&computation_time, &max_computation_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&communication_time, &max_communication_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&boundary_time, &max_boundary_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank == 0) {
            cout << "\nPerformance Statistics:\n"
                 << "Total time: " << max_total_time << " seconds\n"
                 << "Initialization time: " << max_init_time << " seconds\n"
                 << "Computation time: " << max_computation_time << " seconds\n"
                 << "Communication time: " << max_communication_time << " seconds\n"
                 << "Boundary condition time: " << max_boundary_time << " seconds\n"
                 << "Time per step: " << max_total_time / params.K << " seconds\n";
        }

        MPI_Comm_free(&cart_comm);

    } catch (const exception& e) {
        if (rank == 0) {
            cout << "Error: " << e.what() << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}