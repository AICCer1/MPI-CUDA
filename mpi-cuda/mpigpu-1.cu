#define OMPI_SKIP_MPICXX 1

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>

using namespace std;

constexpr double PI = 3.14159265358979323846264338327950288;


__constant__ double d_pi = PI;  
const double h_pi = PI;         


enum Direction {
    EAST = 0,   // X+
    WEST = 1,   // X-
    NORTH = 2,  // Y+
    SOUTH = 3,  // Y-
    FRONT = 4,  // Z+
    BACK = 5    // Z-
};


struct DirInfo {
    int send_idx;      
    int recv_idx;      
    int stride;        
    int size1, size2;   
    int tag_send;      
    int tag_recv;     
};

__device__ __host__ double compute_a_t(double Lx, double Ly, double Lz) {
    #ifdef __CUDA_ARCH__
        return d_pi * sqrt(1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
    #else
        return h_pi * sqrt(1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
    #endif
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


int compute_local_size(int global_size, int num_procs, int proc_coord) {
    int base_size = global_size / num_procs;
    int remainder = global_size % num_procs;
    return base_size + (proc_coord < remainder ? 1 : 0);
}


int compute_start_index(int global_size, int num_procs, int proc_coord) {
    int base_size = global_size / num_procs;
    int remainder = global_size % num_procs;
    return proc_coord * base_size + std::min(proc_coord, remainder);
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

    __host__ __device__ void initialize(const SimulationParams& params, const int coords[3]) {
        this->coords[0] = coords[0];
        this->coords[1] = coords[1];
        this->coords[2] = coords[2];

        local_nx = compute_local_size(params.N - 1, dims[0], coords[0]);
        local_ny = compute_local_size(params.N - 1, dims[1], coords[1]);
        local_nz = compute_local_size(params.N - 1, dims[2], coords[2]);

        start_x = compute_start_index(params.N - 1, dims[0], coords[0]);
        start_y = compute_start_index(params.N - 1, dims[1], coords[1]);
        start_z = compute_start_index(params.N - 1, dims[2], coords[2]);
    }

    void set_boundaries(int N) {
        is_x_left_boundary = (start_x == 0);
        is_x_right_boundary = (start_x + local_nx == N - 1);
        is_y_bottom_boundary = (start_y == 0);
        is_y_top_boundary = (start_y + local_ny == N - 1);
        is_z_front_boundary = (start_z == 0);
        is_z_back_boundary = (start_z + local_nz == N - 1);
    }

    size_t get_buffer_size(int direction) const {
        switch (direction) {
            case EAST: case WEST:  // X
                return (local_ny + 2) * (local_nz + 2);
            case NORTH: case SOUTH:  // Y
                return (local_nx + 2) * (local_nz + 2);
            case FRONT: case BACK:  // Z
                return (local_nx + 2) * (local_ny + 2);
            default:
                throw runtime_error("Invalid direction");
        }
    }
};


inline int compute_index(int i, int j, int k, int local_ny, int local_nz) {
    return i * (local_ny + 2) * (local_nz + 2) + j * (local_nz + 2) + k;
}


__device__ int compute_index_gpu(int i, int j, int k, int local_ny, int local_nz) {
    return i * (local_ny + 2) * (local_nz + 2) + j * (local_nz + 2) + k;
}


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


class ArrayPrinter {
public:
    static void print_mpi_array(const vector<double>& array, 
                              const ProcessGrid& grid,
                              const SimulationParams& params,
                              int rank, int size, MPI_Comm cart_comm,
                              const string& label = "MPI Version") {
        for (int p = 0; p < size; p++) {
            if (rank == p) {
                cout << "\n=== " << label << ": Process " << rank << " ===\n";
                cout << "Process coordinates: (" << grid.coords[0] << ", " 
                     << grid.coords[1] << ", " << grid.coords[2] << ")\n";
                cout << "Local grid size: " << grid.local_nx << "x" 
                     << grid.local_ny << "x" << grid.local_nz << "\n";
                
                for (int i = 0; i <= grid.local_nx + 1; i++) {
                    for (int j = 0; j <= grid.local_ny + 1; j++) {
                        for (int k = 0; k <= grid.local_nz + 1; k++) {
                            int idx = compute_index(i, j, k, grid.local_ny, grid.local_nz);
                            double x = (grid.start_x + i) * params.hx;
                            double y = (grid.start_y + j) * params.hy;
                            double z = (grid.start_z + k) * params.hz;
                            cout << fixed << setprecision(15) 
                                 << "i,j,k=(" << i << "," << j << "," << k << ") "
                                 << "Pos=(" << x << "," << y << "," << z << ") "
                                 << "Value=" << array[idx] << endl;
                        }
                        cout << endl;
                    }
                    cout << "\n----------\n";
                }
                cout << "=====================================\n";
            }
            MPI_Barrier(cart_comm);
            fflush(stdout);
        }
    }

    static void print_cuda_array(double* d_array,
                               const ProcessGrid& grid,
                               const SimulationParams& params,
                               int rank, int size, MPI_Comm cart_comm,
                               const string& label = "MPI+CUDA Version") {
        vector<double> h_array((grid.local_nx + 2) * (grid.local_ny + 2) * (grid.local_nz + 2));
        CUDA_CHECK(cudaMemcpy(h_array.data(), d_array, 
                            h_array.size() * sizeof(double), 
                            cudaMemcpyDeviceToHost));

        print_mpi_array(h_array, grid, params, rank, size, cart_comm, label);
    }
};


struct GPUArrays {
    double *d_u_prev;
    double *d_u_curr;
    double *d_u_next;
    double *d_send_buffers[6];
    double *d_recv_buffers[6];
    
    void allocate(const ProcessGrid& grid) {
        size_t local_size = (grid.local_nx + 2) * (grid.local_ny + 2) * (grid.local_nz + 2);
        CUDA_CHECK(cudaMalloc(&d_u_prev, local_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_u_curr, local_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_u_next, local_size * sizeof(double)));
        
       
        for(int i = 0; i < 6; ++i) {
            size_t buffer_size = grid.get_buffer_size(i);
            CUDA_CHECK(cudaMalloc(&d_send_buffers[i], buffer_size * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_recv_buffers[i], buffer_size * sizeof(double)));
        }
    }
    
    void free() {
        CUDA_CHECK(cudaFree(d_u_prev));
        CUDA_CHECK(cudaFree(d_u_curr));
        CUDA_CHECK(cudaFree(d_u_next));
        
        for(int i = 0; i < 6; ++i) {
            CUDA_CHECK(cudaFree(d_send_buffers[i]));
            CUDA_CHECK(cudaFree(d_recv_buffers[i]));
        }
    }
};

__device__ double compute_u_analytical_gpu(double x, double y, double z, double t, 
                                         const SimulationParams params) {
    return sin(d_pi * x / params.Lx) * 
           sin(d_pi * y / params.Ly) * 
           sin(d_pi * z / params.Lz) * 
           cos(params.a_t * t);
}


vector<DirInfo> create_dir_infos(const ProcessGrid& grid) {
    return vector<DirInfo>{
        // X方向 - east/west
        {grid.local_nx, grid.local_nx + 1, grid.local_nz + 2, grid.local_ny + 2, grid.local_nz + 2, 0, 1},  // EAST
        {1, 0, grid.local_nz + 2, grid.local_ny + 2, grid.local_nz + 2, 1, 0},                              // WEST
        // Y方向 - north/south
        {grid.local_ny, grid.local_ny + 1, grid.local_nz + 2, grid.local_nx + 2, grid.local_nz + 2, 2, 3},  // NORTH
        {1, 0, grid.local_nz + 2, grid.local_nx + 2, grid.local_nz + 2, 3, 2},                              // SOUTH
        // Z方向 - front/back
        {grid.local_nz, grid.local_nz + 1, grid.local_ny + 2, grid.local_nx + 2, grid.local_ny + 2, 4, 5},  // FRONT
        {1, 0, grid.local_ny + 2, grid.local_nx + 2, grid.local_ny + 2, 5, 4}                               // BACK
    };
}


__global__ void prepare_send_buffers_kernel(double *u, double **send_buffers,
                                          const ProcessGrid grid,
                                          const DirInfo *dir_infos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    for(int dir = 0; dir < 6; ++dir) {
        const DirInfo& info = dir_infos[dir];
        int total_size = info.size1 * info.size2;
        if(tid < total_size) {
            int i = tid / info.size2;
            int j = tid % info.size2;
            
            int buffer_idx = i * info.stride + j;
            int u_idx;
            
            if(dir < 2) { // X方向
                u_idx = compute_index_gpu(info.send_idx, i, j, grid.local_ny, grid.local_nz);
            } else if(dir < 4) { // Y方向
                u_idx = compute_index_gpu(i, info.send_idx, j, grid.local_ny, grid.local_nz);
            } else { // Z方向
                u_idx = compute_index_gpu(i, j, info.send_idx, grid.local_ny, grid.local_nz);
            }
            
            send_buffers[dir][buffer_idx] = u[u_idx];
        }
    }
}


__global__ void update_from_recv_buffers_kernel(double *u, double * const *recv_buffers,
                                              const ProcessGrid grid,
                                              const DirInfo *dir_infos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
  
    for(int dir = 0; dir < 6; ++dir) {
        const DirInfo& info = dir_infos[dir];
        int total_size = info.size1 * info.size2;
        if(tid < total_size) {
            int i = tid / info.size2;
            int j = tid % info.size2;
            
            int buffer_idx = i * info.stride + j;
            int u_idx;
            
            if(dir < 2) { 
                u_idx = compute_index_gpu(info.recv_idx, i, j, grid.local_ny, grid.local_nz);
            } else if(dir < 4) { 
                u_idx = compute_index_gpu(i, info.recv_idx, j, grid.local_ny, grid.local_nz);
            } else { 
                u_idx = compute_index_gpu(i, j, info.recv_idx, grid.local_ny, grid.local_nz);
            }
            
            u[u_idx] = recv_buffers[dir][buffer_idx];
        }
    }
}

__global__ void initialize_solution_kernel(double *u, const ProcessGrid grid,
                                         const SimulationParams params, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if(i <= grid.local_nx && j <= grid.local_ny && k <= grid.local_nz) {
        
        double x = (grid.start_x + i) * params.hx;
        double y = (grid.start_y + j) * params.hy;
        double z = (grid.start_z + k) * params.hz;
        
        int idx = compute_index_gpu(i, j, k, grid.local_ny, grid.local_nz);
        u[idx] = compute_u_analytical_gpu(x, y, z, t, params);
    }
}

__global__ void compute_first_timestep_kernel(double *u_next, const double *u_curr,
                                            const ProcessGrid grid, const SimulationParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if(i <= grid.local_nx && j <= grid.local_ny && k <= grid.local_nz) {
        int idx = compute_index_gpu(i, j, k, grid.local_ny, grid.local_nz);
        
        double laplacian = 
            (u_curr[compute_index_gpu(i-1, j, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] + 
             u_curr[compute_index_gpu(i+1, j, k, grid.local_ny, grid.local_nz)]) / (params.hx * params.hx) +
            (u_curr[compute_index_gpu(i, j-1, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] + 
             u_curr[compute_index_gpu(i, j+1, k, grid.local_ny, grid.local_nz)]) / (params.hy * params.hy) +
            (u_curr[compute_index_gpu(i, j, k-1, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] + 
             u_curr[compute_index_gpu(i, j, k+1, grid.local_ny, grid.local_nz)]) / (params.hz * params.hz);
             
        u_next[idx] = u_curr[idx] + 0.5 * params.tau * params.tau * laplacian;
    }
}

__global__ void compute_timestep_kernel(double *u_next, const double *u_curr,
                                      const double *u_prev, const ProcessGrid grid,
                                      const SimulationParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if(i <= grid.local_nx && j <= grid.local_ny && k <= grid.local_nz) {
        int idx = compute_index_gpu(i, j, k, grid.local_ny, grid.local_nz);
        
        double laplacian = 
            (u_curr[compute_index_gpu(i-1, j, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] + 
             u_curr[compute_index_gpu(i+1, j, k, grid.local_ny, grid.local_nz)]) / (params.hx * params.hx) +
            (u_curr[compute_index_gpu(i, j-1, k, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] + 
             u_curr[compute_index_gpu(i, j+1, k, grid.local_ny, grid.local_nz)]) / (params.hy * params.hy) +
            (u_curr[compute_index_gpu(i, j, k-1, grid.local_ny, grid.local_nz)] - 2.0 * u_curr[idx] + 
             u_curr[compute_index_gpu(i, j, k+1, grid.local_ny, grid.local_nz)]) / (params.hz * params.hz);
             
        u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + params.tau * params.tau * laplacian;
    }
}

__global__ void apply_boundary_conditions_kernel(double *u, const ProcessGrid grid) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = grid.local_nx + 2;
    int ny = grid.local_ny + 2;
    int nz = grid.local_nz + 2;

    // X
    if (grid.is_x_left_boundary) {
        int total_size = ny * nz;
        if (tid < total_size) {
            int j = tid / nz;
            int k = tid % nz;
            int idx = compute_index_gpu(0, j, k, grid.local_ny, grid.local_nz);
            u[idx] = 0.0;
        }
    }
    if (grid.is_x_right_boundary) {
        int total_size = ny * nz;
        if (tid < total_size) {
            int j = tid / nz;
            int k = tid % nz;
            int idx = compute_index_gpu(grid.local_nx + 1, j, k, grid.local_ny, grid.local_nz);
            u[idx] = 0.0;
        }
    }

    // Y
    if (grid.is_y_bottom_boundary) {
        int total_size = nx * nz;
        if (tid < total_size) {
            int i = tid / nz;
            int k = tid % nz;
            int idx = compute_index_gpu(i, 0, k, grid.local_ny, grid.local_nz);
            u[idx] = 0.0;
        }
    }
    if (grid.is_y_top_boundary) {
        int total_size = nx * nz;
        if (tid < total_size) {
            int i = tid / nz;
            int k = tid % nz;
            int idx = compute_index_gpu(i, grid.local_ny + 1, k, grid.local_ny, grid.local_nz);
            u[idx] = 0.0;
        }
    }

    // Z
    if (grid.is_z_front_boundary) {
        int total_size = nx * ny;
        if (tid < total_size) {
            int i = tid / ny;
            int j = tid % ny;
            int idx = compute_index_gpu(i, j, 0, grid.local_ny, grid.local_nz);
            u[idx] = 0.0;
        }
    }
    if (grid.is_z_back_boundary) {
        int total_size = nx * ny;
        if (tid < total_size) {
            int i = tid / ny;
            int j = tid % ny;
            int idx = compute_index_gpu(i, j, grid.local_nz + 1, grid.local_ny, grid.local_nz);
            u[idx] = 0.0;
        }
    }
}

void exchange_boundaries_3d_cuda(GPUArrays &gpu_arrays, double *current_u,
                               vector<vector<double>> &host_send_buffers,
                               vector<vector<double>> &host_recv_buffers,
                               const ProcessGrid& grid, MPI_Comm cart_comm,
                               double **d_send_buffers_ptr,
                               double **d_recv_buffers_ptr,vector<DirInfo> host_dir_infos,DirInfo *d_dir_infos,int neighbors[6]) {
    
 
    CUDA_CHECK(cudaMemcpy(d_send_buffers_ptr, gpu_arrays.d_send_buffers,
                         6 * sizeof(double*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_recv_buffers_ptr, gpu_arrays.d_recv_buffers,
                         6 * sizeof(double*), cudaMemcpyHostToDevice));

  
    int max_threads = 0;
    for (int dir = 0; dir < 6; ++dir) {
        int size = host_dir_infos[dir].size1 * host_dir_infos[dir].size2;
        if (size > max_threads) max_threads = size;
    }
    int threads_per_block = 256;
    int num_blocks = (max_threads + threads_per_block - 1) / threads_per_block;

  
    prepare_send_buffers_kernel<<<num_blocks, threads_per_block>>>(
        current_u, d_send_buffers_ptr, grid, d_dir_infos);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<MPI_Request> requests;
    requests.reserve(12);

  
    for(int dir = 0; dir < 6; ++dir) {
        if(neighbors[dir] != MPI_PROC_NULL) {
            size_t buffer_size = grid.get_buffer_size(dir);
            
            CUDA_CHECK(cudaMemcpy(host_send_buffers[dir].data(),
                                gpu_arrays.d_send_buffers[dir],
                                buffer_size * sizeof(double),
                                cudaMemcpyDeviceToHost));

            MPI_Request send_request, recv_request;
            MPI_Isend(host_send_buffers[dir].data(), buffer_size, MPI_DOUBLE,
                     neighbors[dir], host_dir_infos[dir].tag_send, cart_comm, &send_request);
            MPI_Irecv(host_recv_buffers[dir].data(), buffer_size, MPI_DOUBLE,
                     neighbors[dir], host_dir_infos[dir].tag_recv, cart_comm, &recv_request);
            
            requests.push_back(send_request);
            requests.push_back(recv_request);
        }
    }

   
    if(!requests.empty()) {
        vector<MPI_Status> statuses(requests.size());
        MPI_Waitall(requests.size(), requests.data(), statuses.data());
    }


    for(int dir = 0; dir < 6; ++dir) {
        if(neighbors[dir] != MPI_PROC_NULL) {
            size_t buffer_size = grid.get_buffer_size(dir);
            CUDA_CHECK(cudaMemcpy(gpu_arrays.d_recv_buffers[dir],
                                host_recv_buffers[dir].data(),
                                buffer_size * sizeof(double),
                                cudaMemcpyHostToDevice));
        }
    }

   
    update_from_recv_buffers_kernel<<<num_blocks, threads_per_block>>>(
        current_u, d_recv_buffers_ptr, grid, d_dir_infos);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}


struct Error {
    double max_error = 0.0;
    double l2_error = 0.0;
    size_t count = 0;

    void clear() {
        max_error = l2_error = 0.0;
        count = 0;
    }

    void update(double error) {
        max_error = max(max_error, fabs(error));
        l2_error += error * error;
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


__device__ void atomicMax(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

__global__ void compute_error_kernel(const double *u_numerical, const ProcessGrid grid,
                                   const SimulationParams params, double t,
                                   double *max_error, double *l2_error_array) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if(i <= grid.local_nx && j <= grid.local_ny && k <= grid.local_nz) {
        double x = (grid.start_x + i) * params.hx;
        double y = (grid.start_y + j) * params.hy;
        double z = (grid.start_z + k) * params.hz;
        
        int idx = compute_index_gpu(i, j, k, grid.local_ny, grid.local_nz);
        double u_exact = compute_u_analytical_gpu(x, y, z, t, params);
        double error = fabs(u_numerical[idx] - u_exact);
        
        atomicMax(max_error, error); 
        l2_error_array[idx] = error * error;
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

        
        auto total_start_time = MPI_Wtime();
        auto start_time = MPI_Wtime();

     
        if(argc < 5) {
            if(rank == 0) {
                cout << "Usage: " << argv[0] << " N tau K number_threads\n";
            }
            MPI_Finalize();
            return 1;
        }

     
        SimulationParams params;
        params.initialize(
            stoi(argv[1]),    // N
            stod(argv[2]),    // tau
            stoi(argv[3])     // K
        );
        int number_threads = stoi(argv[4]);

    
        omp_set_num_threads(number_threads);

     
        ProcessGrid grid;
        grid.dims[0] = grid.dims[1] = grid.dims[2] = 0;
        MPI_Dims_create(size, 3, grid.dims);

     
        int periods[3] = {0, 0, 0};
        MPI_Comm cart_comm;
        int cart_create_result = MPI_Cart_create(MPI_COMM_WORLD, 3, grid.dims, periods, 0, &cart_comm);
        if (cart_create_result != MPI_SUCCESS) {
            throw runtime_error("Failed to create cartesian communicator");
        }

        MPI_Cart_coords(cart_comm, rank, 3, grid.coords);

       
        grid.initialize(params, grid.coords);
        grid.set_boundaries(params.N);

       
        char* local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        int local_rank = 0;
        
        if (local_rank_str != nullptr) {
            local_rank = atoi(local_rank_str);
        } else {
            
            MPI_Comm node_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, 
                               MPI_INFO_NULL, &node_comm);
            MPI_Comm_rank(node_comm, &local_rank);
            MPI_Comm_free(&node_comm);
        }
        
        
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        int target_device = local_rank % device_count;
        CUDA_CHECK(cudaSetDevice(target_device));

      
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, target_device));
        
       
        for (int p = 0; p < size; p++) {
            if (p == rank) {
                if (rank == 0) {
                    cout << "Configuration:\n"
                         << "Grid size: " << params.N << "x" << params.N << "x" << params.N << "\n"
                         << "Process grid: " << grid.dims[0] << "x" << grid.dims[1] << "x" << grid.dims[2] << "\n"
                         << "Local grid size: " << grid.local_nx << "x" << grid.local_ny << "x" << grid.local_nz << "\n"
                         << "Time steps: " << params.K << "\n"
                         << "Threads per process: " << number_threads << "\n\n";
                }
                cout << "Process " << rank 
                     << " (Local Rank " << local_rank 
                     << ") using GPU " << target_device 
                     << ": " << prop.name << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
     
        GPUArrays gpu_arrays;
        gpu_arrays.allocate(grid);

      
        vector<vector<double>> host_send_buffers(6);
        vector<vector<double>> host_recv_buffers(6);
        for(int i = 0; i < 6; ++i) {
            size_t buffer_size = grid.get_buffer_size(i);
            host_send_buffers[i].resize(buffer_size);
            host_recv_buffers[i].resize(buffer_size);
        }

        double **d_send_buffers_ptr, **d_recv_buffers_ptr;
        CUDA_CHECK(cudaMalloc(&d_send_buffers_ptr, 6 * sizeof(double*)));
        CUDA_CHECK(cudaMalloc(&d_recv_buffers_ptr, 6 * sizeof(double*)));

     
        DirInfo *d_dir_infos;
        CUDA_CHECK(cudaMalloc(&d_dir_infos, 6 * sizeof(DirInfo)));
     
        vector<DirInfo> host_dir_infos = create_dir_infos(grid);
        
        CUDA_CHECK(cudaMemcpy(d_dir_infos, host_dir_infos.data(), 
                         6 * sizeof(DirInfo), cudaMemcpyHostToDevice));

  
        int neighbors[6];
        MPI_Cart_shift(cart_comm, 0, 1, &neighbors[WEST], &neighbors[EAST]);   // X
        MPI_Cart_shift(cart_comm, 1, 1, &neighbors[SOUTH], &neighbors[NORTH]); // Y
        MPI_Cart_shift(cart_comm, 2, 1, &neighbors[BACK], &neighbors[FRONT]);  // Z

     
        dim3 block_size(8, 8, 8);
        dim3 grid_size(
            (grid.local_nx + block_size.x - 1) / block_size.x,
            (grid.local_ny + block_size.y - 1) / block_size.y,
            (grid.local_nz + block_size.z - 1) / block_size.z
        );

      
        initialize_solution_kernel<<<grid_size, block_size>>>(
            gpu_arrays.d_u_prev, grid, params, 0.0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        init_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        int threads_per_block = 256;
        int max_threads = max(max((grid.local_ny + 2) * (grid.local_nz + 2),
                                 (grid.local_nx + 2) * (grid.local_nz + 2)),
                             (grid.local_nx + 2) * (grid.local_ny + 2));
        int num_blocks = (max_threads + threads_per_block - 1) / threads_per_block;
        
        apply_boundary_conditions_kernel<<<num_blocks, threads_per_block>>>(
            gpu_arrays.d_u_prev, grid);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        boundary_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        exchange_boundaries_3d_cuda(gpu_arrays, gpu_arrays.d_u_prev,
                                  host_send_buffers, host_recv_buffers,
                                  grid, cart_comm, d_send_buffers_ptr, 
                                  d_recv_buffers_ptr,host_dir_infos,d_dir_infos,neighbors);
        
        communication_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

      
        compute_first_timestep_kernel<<<grid_size, block_size>>>(
            gpu_arrays.d_u_curr, gpu_arrays.d_u_prev, grid, params);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        computation_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        apply_boundary_conditions_kernel<<<num_blocks, threads_per_block>>>(
            gpu_arrays.d_u_curr, grid);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        boundary_time += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();

        exchange_boundaries_3d_cuda(gpu_arrays, gpu_arrays.d_u_curr,
                                  host_send_buffers, host_recv_buffers,
                                  grid, cart_comm,d_send_buffers_ptr, 
                                  d_recv_buffers_ptr,host_dir_infos,d_dir_infos,neighbors);

        communication_time += MPI_Wtime() - start_time;

     
        for(int n = 2; n <= params.K; ++n) {

            double t = n * params.tau;
            start_time = MPI_Wtime();
            
          
            compute_timestep_kernel<<<grid_size, block_size>>>(
                gpu_arrays.d_u_next, gpu_arrays.d_u_curr, gpu_arrays.d_u_prev,
                grid, params);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            computation_time += MPI_Wtime() - start_time;
            start_time = MPI_Wtime();

           
            apply_boundary_conditions_kernel<<<num_blocks, threads_per_block>>>(
                gpu_arrays.d_u_next, grid);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            boundary_time += MPI_Wtime() - start_time;
            start_time = MPI_Wtime();

       
            exchange_boundaries_3d_cuda(gpu_arrays, gpu_arrays.d_u_next,
                                      host_send_buffers, host_recv_buffers,
                                      grid, cart_comm,d_send_buffers_ptr, 
                                      d_recv_buffers_ptr,host_dir_infos,d_dir_infos,neighbors);
            
            communication_time += MPI_Wtime() - start_time;

                    
            // if(n % (params.K / 10) == 0 || n == params.K) {
            //     double *d_max_error, *d_l2_error;
            //     size_t local_size = (grid.local_nx + 2) * (grid.local_ny + 2) * (grid.local_nz + 2);
            //     CUDA_CHECK(cudaMalloc(&d_max_error, sizeof(double)));
            //     CUDA_CHECK(cudaMalloc(&d_l2_error, local_size * sizeof(double)));

            //     CUDA_CHECK(cudaMemset(d_max_error, 0, sizeof(double)));
            //     CUDA_CHECK(cudaMemset(d_l2_error, 0, local_size * sizeof(double)));

            //     compute_error_kernel<<<grid_size, block_size>>>(
            //         gpu_arrays.d_u_next, grid, params, t, d_max_error, d_l2_error);
            //     CUDA_CHECK(cudaGetLastError());
            //     CUDA_CHECK(cudaDeviceSynchronize());

            //     Error error;
            //     double max_error_host;
            //     vector<double> l2_errors(local_size);
                
            //     CUDA_CHECK(cudaMemcpy(&max_error_host, d_max_error, sizeof(double),
            //                         cudaMemcpyDeviceToHost));
            //     CUDA_CHECK(cudaMemcpy(l2_errors.data(), d_l2_error,
            //                         local_size * sizeof(double),
            //                         cudaMemcpyDeviceToHost));

            //     double l2_error_host = 0.0;
            //     for(size_t i = 0; i < local_size; ++i) {
            //         l2_error_host += l2_errors[i];
            //     }

            //     error.max_error = max_error_host;
            //     error.l2_error = l2_error_host;
            //     error.count = grid.local_nx * grid.local_ny * grid.local_nz;

            //     error.reduce(cart_comm);

            //     if(rank == 0) {
            //         cout << "Step " << n 
            //              << ", t = " << fixed << setprecision(6) << t
            //              << ", Max Error = " << scientific << setprecision(6) 
            //              << error.get_max_error()
            //              << ", L2 Error = " << error.get_l2_error() << endl;
            //     }

            //     CUDA_CHECK(cudaFree(d_max_error));
            //     CUDA_CHECK(cudaFree(d_l2_error));
            // }

          
            swap(gpu_arrays.d_u_prev, gpu_arrays.d_u_curr);
            swap(gpu_arrays.d_u_curr, gpu_arrays.d_u_next);
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

        
        gpu_arrays.free();
        
      
        CUDA_CHECK(cudaFree(d_dir_infos));
        CUDA_CHECK(cudaFree(d_send_buffers_ptr));
        CUDA_CHECK(cudaFree(d_recv_buffers_ptr));
        
        MPI_Comm_free(&cart_comm);

    } catch (const exception& e) {
        if(rank == 0) {
            cout << "Error: " << e.what() << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}