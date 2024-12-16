#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
using namespace std;

const double pi = acos(-1.0);

inline int compute_index(int i, int j, int k, int N) {
    return i * (N + 1) * (N + 1) + j * (N + 1) + k;
}

double compute_a_t(double Lx, double Ly, double Lz) {
    return pi * sqrt(1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
}

double compute_u_analytical(double x, double y, double z, double t, double Lx, double Ly, double Lz, double a_t) {
    return sin(pi * x / Lx) * sin(pi * y / Ly) * sin(pi * z / Lz) * cos(a_t * t);
}

struct Error {
    double max_error = 0.0;
    double l2_error = 0.0;
    size_t count = 0;

    void print(int step, double time) const {
        cout << "Step " << step 
             << ", t = " << fixed << setprecision(6) << time
             << ", Max Error = " << scientific << setprecision(6) << max_error 
             << ", L2 Error = " << l2_error << endl;
    }
};

void apply_boundary_conditions(vector<double>& u, int N) {
   
    for(int j = 0; j <= N; ++j) {
        for(int k = 0; k <= N; ++k) {
            u[compute_index(0, j, k, N)] = 0.0;
            u[compute_index(N, j, k, N)] = 0.0;
        }
    }

    for(int i = 0; i <= N; ++i) {
        for(int k = 0; k <= N; ++k) {
            u[compute_index(i, 0, k, N)] = 0.0;
            u[compute_index(i, N, k, N)] = 0.0;
        }
    }

   
    for(int i = 0; i <= N; ++i) {
        for(int j = 0; j <= N; ++j) {
            u[compute_index(i, j, 0, N)] = 0.0;
            u[compute_index(i, j, N, N)] = 0.0;
        }
    }
}

Error compute_error(const vector<double>& u_numerical, int N, double t, 
                   double Lx, double Ly, double Lz, double a_t,
                   double hx, double hy, double hz) {
    double max_error = 0.0;
    double l2_error = 0.0;
    size_t count = 0;
    
    
    for(int i = 1; i < N; ++i) {
        for(int j = 1; j < N; ++j) {
            for(int k = 1; k < N; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                
                double u_exact = compute_u_analytical(x, y, z, t, Lx, Ly, Lz, a_t);
                int idx = compute_index(i, j, k, N);
                double local_error = fabs(u_numerical[idx] - u_exact);
                
                max_error = max(max_error, local_error);
                l2_error += local_error * local_error;
                count++;
            }
        }
    }
    
    Error error;
    error.max_error = max_error;
    error.l2_error = sqrt(l2_error / count);
    // error.l2_error = sqrt(l2_error *hx*hy*hz);

    error.count = count;
    return error;
}

int main(int argc, char* argv[]) {
    
    double init_time = 0.0;
    double computation_time = 0.0;
    double boundary_time = 0.0;

  
    auto total_start_time = clock();
    auto start_time = clock();

    const double Lx = 1.0;  
    const double Ly = 1.0;
    const double Lz = 1.0;

    if(argc < 4){
        cout << "Usage: " << argv[0] << " N tau K\n";
        return 1;
    }

    int N = stoi(argv[1]);
    double tau = stod(argv[2]);
    int K = stoi(argv[3]);
    
    double T = tau * K;
    double hx = Lx / N;
    double hy = Ly / N;
    double hz = Lz / N;
    double a_t = compute_a_t(Lx, Ly, Lz);

    cout << "Configuration:\n"
         << "Grid size: " << N << "x" << N << "x" << N << "\n"
         << "Time steps: " << K << "\n\n";

    size_t total_points = static_cast<size_t>(N + 1) * (N + 1) * (N + 1);
    vector<double> u_prev(total_points, 0.0);
    vector<double> u_curr(total_points, 0.0);
    vector<double> u_next(total_points, 0.0);

    init_time += double(clock() - start_time) / CLOCKS_PER_SEC;
    start_time = clock();

  
    for(int i = 0; i <= N; ++i) {
        for(int j = 0; j <= N; ++j) {
            for(int k = 0; k <= N; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                int idx = compute_index(i, j, k, N);
                u_prev[idx] = compute_u_analytical(x, y, z, 0.0, Lx, Ly, Lz, a_t);
            }
        }
    }
    computation_time += double(clock() - start_time) / CLOCKS_PER_SEC;
    start_time = clock();

    apply_boundary_conditions(u_prev, N);
    boundary_time += double(clock() - start_time) / CLOCKS_PER_SEC;
    start_time = clock();

   
    for(int i = 1; i < N; ++i) {
        for(int j = 1; j < N; ++j) {
            for(int k = 1; k < N; ++k) {
                int idx = compute_index(i, j, k, N);
                double laplacian = 
                    (u_prev[compute_index(i-1, j, k, N)] - 2.0 * u_prev[idx] + u_prev[compute_index(i+1, j, k, N)]) / (hx * hx) +
                    (u_prev[compute_index(i, j-1, k, N)] - 2.0 * u_prev[idx] + u_prev[compute_index(i, j+1, k, N)]) / (hy * hy) +
                    (u_prev[compute_index(i, j, k-1, N)] - 2.0 * u_prev[idx] + u_prev[compute_index(i, j, k+1, N)]) / (hz * hz);
                
                u_curr[idx] = u_prev[idx] + 0.5 * tau * tau * laplacian;
            }
        }
    }
    computation_time += double(clock() - start_time) / CLOCKS_PER_SEC;
    start_time = clock();

    apply_boundary_conditions(u_curr, N);
    boundary_time += double(clock() - start_time) / CLOCKS_PER_SEC;

   
    for(int n = 2; n <= K; ++n) {
        double t = n * tau;
        
        start_time = clock();
        
        for(int i = 1; i < N; ++i) {
            for(int j = 1; j < N; ++j) {
                for(int k = 1; k < N; ++k) {
                    int idx = compute_index(i, j, k, N);
                    double laplacian = 
                        (u_curr[compute_index(i-1, j, k, N)] - 2.0 * u_curr[idx] + u_curr[compute_index(i+1, j, k, N)]) / (hx * hx) +
                        (u_curr[compute_index(i, j-1, k, N)] - 2.0 * u_curr[idx] + u_curr[compute_index(i, j+1, k, N)]) / (hy * hy) +
                        (u_curr[compute_index(i, j, k-1, N)] - 2.0 * u_curr[idx] + u_curr[compute_index(i, j, k+1, N)]) / (hz * hz);
                    
                    u_next[idx] = 2.0 * u_curr[idx] - u_prev[idx] + tau * tau * laplacian;
                }
            }
        }
        computation_time += double(clock() - start_time) / CLOCKS_PER_SEC;
        start_time = clock();

        apply_boundary_conditions(u_next, N);
        boundary_time += double(clock() - start_time) / CLOCKS_PER_SEC;

        if(n % (K / 10) == 0 || n == K) {
            start_time = clock();
            Error error = compute_error(u_next, N, t, Lx, Ly, Lz, a_t, hx, hy, hz);
            computation_time += double(clock() - start_time) / CLOCKS_PER_SEC;
            error.print(n, t);
        }

        swap(u_prev, u_curr);
        swap(u_curr, u_next);
    }

    double total_time = double(clock() - total_start_time) / CLOCKS_PER_SEC;

    cout << "\nDetailed Performance Statistics:\n"
         << "Total time: " << total_time << " seconds\n"
         << "Initialization time: " << init_time << " seconds\n"
         << "Computation time: " << computation_time << " seconds\n"
         << "Boundary condition time: " << boundary_time << " seconds\n"
         << "Time per step: " << total_time / K << " seconds\n";

    return 0;
}