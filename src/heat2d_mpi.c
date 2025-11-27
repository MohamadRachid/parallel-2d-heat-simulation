#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX 200
#define NY 200
#define NSTEPS 1000

// Access macro for 1D array storing 2D grid with halos
#define IDX(i,j,ny) ((i)*(ny) + (j))

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 2D Cartesian grid
    int dims[2]    = {0, 0};
    int periods[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    if (rank == 0) {
        printf("Total MPI processes: %d\n", size);
        printf("Process grid dims: %d x %d\n", dims[0], dims[1]);
    }

    MPI_Comm cart_comm;
    MPI_Cart_create(comm, 2, dims, periods, 1, &cart_comm);

    if (cart_comm == MPI_COMM_NULL) {
        if (rank == 0) fprintf(stderr, "Error: MPI_Cart_create failed\n");
        MPI_Finalize();
        return 1;
    }

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int px = dims[0];  // number of process rows
    int py = dims[1];  // number of process cols

    // Local domain size (assuming evenly divisible)
    int local_nx = NX / px;  // local number of rows
    int local_ny = NY / py;  // local number of cols

    // Full local sizes including halos
    int nx = local_nx + 2;   // i = 0 .. local_nx+1
    int ny = local_ny + 2;   // j = 0 .. local_ny+1

    // Neighbor ranks
    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);

    // Physical parameters
    double alpha = 1.0;
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    double dt = 0.2 * dx * dx / alpha; // stable

    if (rank == 0) {
        printf("dx = %g, dy = %g, dt = %g\n", dx, dy, dt);
    }

    // Global start indices (for interior cell i=1,j=1)
    int global_i_start = coords[0] * local_nx;
    int global_j_start = coords[1] * local_ny;
    int global_i_end   = global_i_start + local_nx - 1;
    int global_j_end   = global_j_start + local_ny - 1;

    // Allocate local arrays with halos
    size_t total_size = (size_t)nx * (size_t)ny;
    double *u     = (double*)malloc(total_size * sizeof(double));
    double *u_new = (double*)malloc(total_size * sizeof(double));
    if (!u || !u_new) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(cart_comm, 1);
    }

    // Initialize: everything zero
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            u[IDX(i,j,ny)] = u_new[IDX(i,j,ny)] = 0.0;

    // Initial hot square in the middle of the global domain
    int istart_hot = (int)(NX * 0.3);
    int iend_hot   = (int)(NX * 0.7);
    int jstart_hot = (int)(NY * 0.3);
    int jend_hot   = (int)(NY * 0.7);

    for (int i = 1; i <= local_nx; ++i) {
        int gi = global_i_start + (i - 1);
        for (int j = 1; j <= local_ny; ++j) {
            int gj = global_j_start + (j - 1);
            if (gi >= istart_hot && gi < iend_hot &&
                gj >= jstart_hot && gj < jend_hot) {
                u[IDX(i,j,ny)] = 1.0;
            } else {
                u[IDX(i,j,ny)] = 0.0;
            }
        }
    }

    // Buffers for east/west columns
    double *send_west  = NULL, *recv_west = NULL;
    double *send_east  = NULL, *recv_east = NULL;
    if (west != MPI_PROC_NULL || east != MPI_PROC_NULL) {
        send_west =  (double*)malloc(local_nx * sizeof(double));
        recv_west =  (double*)malloc(local_nx * sizeof(double));
        send_east =  (double*)malloc(local_nx * sizeof(double));
        recv_east =  (double*)malloc(local_nx * sizeof(double));
        if (!send_west || !recv_west || !send_east || !recv_east) {
            fprintf(stderr, "Rank %d: buffer malloc failed\n", rank);
            MPI_Abort(cart_comm, 1);
        }
    }

    // =========================
    // TIMING START
    // =========================
    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    // Time stepping
    for (int n = 0; n < NSTEPS; ++n) {

        // 1) HALO EXCHANGE
        MPI_Request reqs[8];
        int nreq = 0;

        // --- North/South rows (contiguous) ---
        if (north != MPI_PROC_NULL) {
            MPI_Irecv(&u[IDX(0,1,ny)], local_ny, MPI_DOUBLE,
                      north, 0, cart_comm, &reqs[nreq++]);
            MPI_Isend(&u[IDX(1,1,ny)], local_ny, MPI_DOUBLE,
                      north, 1, cart_comm, &reqs[nreq++]);
        } else {
            for (int j = 1; j <= local_ny; ++j)
                u[IDX(0,j,ny)] = 0.0;
        }

        if (south != MPI_PROC_NULL) {
            MPI_Irecv(&u[IDX(local_nx+1,1,ny)], local_ny, MPI_DOUBLE,
                      south, 1, cart_comm, &reqs[nreq++]);
            MPI_Isend(&u[IDX(local_nx,1,ny)], local_ny, MPI_DOUBLE,
                      south, 0, cart_comm, &reqs[nreq++]);
        } else {
            for (int j = 1; j <= local_ny; ++j)
                u[IDX(local_nx+1,j,ny)] = 0.0;
        }

        // --- West/East columns (need packing) ---
        if (west != MPI_PROC_NULL) {
            for (int i = 1; i <= local_nx; ++i)
                send_west[i-1] = u[IDX(i,1,ny)];

            MPI_Irecv(recv_west, local_nx, MPI_DOUBLE,
                      west, 2, cart_comm, &reqs[nreq++]);
            MPI_Isend(send_west, local_nx, MPI_DOUBLE,
                      west, 3, cart_comm, &reqs[nreq++]);
        } else {
            for (int i = 1; i <= local_nx; ++i)
                u[IDX(i,0,ny)] = 0.0;
        }

        if (east != MPI_PROC_NULL) {
            for (int i = 1; i <= local_nx; ++i)
                send_east[i-1] = u[IDX(i,local_ny,ny)];

            MPI_Irecv(recv_east, local_nx, MPI_DOUBLE,
                      east, 3, cart_comm, &reqs[nreq++]);
            MPI_Isend(send_east, local_nx, MPI_DOUBLE,
                      east, 2, cart_comm, &reqs[nreq++]);
        } else {
            for (int i = 1; i <= local_nx; ++i)
                u[IDX(i,local_ny+1,ny)] = 0.0;
        }

        if (nreq > 0)
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Unpack west/east halos
        if (west != MPI_PROC_NULL)
            for (int i = 1; i <= local_nx; ++i)
                u[IDX(i,0,ny)] = recv_west[i-1];

        if (east != MPI_PROC_NULL)
            for (int i = 1; i <= local_nx; ++i)
                u[IDX(i,local_ny+1,ny)] = recv_east[i-1];

        // 2) ENFORCE DIRICHLET BOUNDARY ON PHYSICAL EDGES
        if (global_i_start == 0) {
            int i = 1;
            for (int j = 1; j <= local_ny; ++j)
                u[IDX(i,j,ny)] = 0.0;
        }
        if (global_i_end == NX-1) {
            int i = local_nx;
            for (int j = 1; j <= local_ny; ++j)
                u[IDX(i,j,ny)] = 0.0;
        }
        if (global_j_start == 0) {
            int j = 1;
            for (int i = 1; i <= local_nx; ++i)
                u[IDX(i,j,ny)] = 0.0;
        }
        if (global_j_end == NY-1) {
            int j = local_ny;
            for (int i = 1; i <= local_nx; ++i)
                u[IDX(i,j,ny)] = 0.0;
        }

        // 3) UPDATE INTERIOR POINTS
        for (int i = 1; i <= local_nx; ++i) {
            for (int j = 1; j <= local_ny; ++j) {
                int gi = global_i_start + (i - 1);
                int gj = global_j_start + (j - 1);
                if (gi == 0 || gi == NX-1 || gj == 0 || gj == NY-1) {
                    u_new[IDX(i,j,ny)] = 0.0;
                    continue;
                }

                double uij  = u[IDX(i,j,ny)];
                double uip1 = u[IDX(i+1,j,ny)];
                double uim1 = u[IDX(i-1,j,ny)];
                double ujp1 = u[IDX(i,j+1,ny)];
                double ujm1 = u[IDX(i,j-1,ny)];

                double dudx2 = (uip1 - 2.0*uij + uim1) / (dx*dx);
                double dudy2 = (ujp1 - 2.0*uij + ujm1) / (dy*dy);

                u_new[IDX(i,j,ny)] = uij + dt * alpha * (dudx2 + dudy2);
            }
        }

        // Swap u and u_new
        double *tmp = u; u = u_new; u_new = tmp;

        if ((n+1) % 100 == 0 && rank == 0)
            printf("Step %d / %d\n", n+1, NSTEPS);
    }

    // =========================
    // TIMING END
    // =========================
    MPI_Barrier(cart_comm);
    double t_end = MPI_Wtime();
    if (rank == 0)
        printf("\nTotal simulation time: %.6f seconds\n", t_end - t_start);

    // ============================================
    // GLOBAL HEAT SUM (correctness diagnostic)
    // ============================================
    double local_sum = 0.0;
    for (int i = 1; i <= local_nx; ++i)
        for (int j = 1; j <= local_ny; ++j)
            local_sum += u[IDX(i,j,ny)];

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, cart_comm);

    if (rank == 0)
        printf("\nGlobal heat sum after %d steps = %.12f\n",
               NSTEPS, global_sum);

    // ============================================
    // GATHER FULL FIELD TO RANK 0 FOR CSV OUTPUT
    // (FIXED VERSION: use MPI_Gather + manual placement)
    // ============================================
    int local_count = local_nx * local_ny;
    double *localbuf = (double*)malloc(local_count * sizeof(double));
    if (!localbuf) {
        fprintf(stderr, "Rank %d: localbuf malloc failed\n", rank);
        MPI_Abort(cart_comm, 1);
    }

    int idx = 0;
    for (int i = 1; i <= local_nx; ++i)
        for (int j = 1; j <= local_ny; ++j)
            localbuf[idx++] = u[IDX(i,j,ny)];

    double *all_local = NULL;
    double *global_field = NULL;

    if (rank == 0) {
        all_local   = (double*)malloc((size_t)local_count * size * sizeof(double));
        global_field = (double*)malloc((size_t)NX * NY * sizeof(double));
        if (!all_local || !global_field) {
            fprintf(stderr, "Rank 0: malloc for gather/global_field failed\n");
            MPI_Abort(cart_comm, 1);
        }
        // Initialize so we don't get garbage
        for (int i = 0; i < NX*NY; ++i)
            global_field[i] = 0.0;
    }

    // Gather all local blocks as a flat array on rank 0
    MPI_Gather(localbuf, local_count, MPI_DOUBLE,
               all_local, local_count, MPI_DOUBLE,
               0, cart_comm);

    if (rank == 0) {
        // For each rank r, copy its block into the right place in global_field
        for (int r = 0; r < size; ++r) {
            int ccoords[2];
            MPI_Cart_coords(cart_comm, r, 2, ccoords);
            int gi0 = ccoords[0] * local_nx; // starting global row
            int gj0 = ccoords[1] * local_ny; // starting global col

            int base = r * local_count; // offset into all_local

            for (int li = 0; li < local_nx; ++li) {
                for (int lj = 0; lj < local_ny; ++lj) {
                    double val = all_local[base + li*local_ny + lj];
                    int gi = gi0 + li;
                    int gj = gj0 + lj;
                    global_field[gi * NY + gj] = val;
                }
            }
        }

        // Write CSV
        FILE *f = fopen("u_mpi_final.csv", "w");
        if (!f) {
            fprintf(stderr, "Rank 0: failed to open u_mpi_final.csv for writing\n");
        } else {
            for (int i = 0; i < NX; ++i) {
                for (int j = 0; j < NY; ++j) {
                    fprintf(f, "%.6f", global_field[i*NY + j]);
                    if (j < NY-1) fprintf(f, ",");
                }
                fprintf(f, "\n");
            }
            fclose(f);
            printf("\nWrote output: u_mpi_final.csv\n");
        }
    }

    // Cleanup
    free(localbuf);
    if (rank == 0) {
        free(all_local);
        free(global_field);
    }

    if (send_west) {
        free(send_west);
        free(recv_west);
        free(send_east);
        free(recv_east);
    }

    free(u);
    free(u_new);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

