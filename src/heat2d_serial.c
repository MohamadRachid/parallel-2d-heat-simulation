#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 100
#define NY 100

#define NSTEPS 1000

// Access macro for 1D array storing 2D grid
#define IDX(i,j) ((i) * NY + (j))

void initialize(double *u) {
    // Set everything to 0
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            u[IDX(i,j)] = 0.0;
        }
    }

    // Hot square in the middle: 40% x 40% of the domain
    int istart = NX * 0.3;
    int iend   = NX * 0.7;
    int jstart = NY * 0.3;
    int jend   = NY * 0.7;

    for (int i = istart; i < iend; ++i) {
        for (int j = jstart; j < jend; ++j) {
            u[IDX(i,j)] = 1.0;
        }
    }
}

void apply_dirichlet_boundaries(double *u) {
    // All boundaries = 0
    // Top and bottom rows
    for (int j = 0; j < NY; ++j) {
        u[IDX(0, j)]      = 0.0; // top
        u[IDX(NX-1, j)]   = 0.0; // bottom
    }
    // Left and right columns
    for (int i = 0; i < NX; ++i) {
        u[IDX(i, 0)]      = 0.0; // left
        u[IDX(i, NY-1)]   = 0.0; // right
    }
}

int main(void) {
    double alpha = 1.0;

    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);

    // For dx = dy, stable if dt <= dx^2 / (4 * alpha).
    double dt = 0.2 * dx * dx / alpha; // safety factor 0.2

    printf("NX = %d, NY = %d\n", NX, NY);
    printf("dx = %g, dy = %g, dt = %g\n", dx, dy, dt);

    size_t size = (size_t)NX * (size_t)NY;

    double *u     = (double*)malloc(size * sizeof(double));
    double *u_new = (double*)malloc(size * sizeof(double));

    if (!u || !u_new) {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }

    initialize(u);
    apply_dirichlet_boundaries(u);

    // Time stepping
    for (int n = 0; n < NSTEPS; ++n) {

        // Update interior points (1 .. NX-2, 1 .. NY-2)
        for (int i = 1; i < NX-1; ++i) {
            for (int j = 1; j < NY-1; ++j) {
                double uij  = u[IDX(i,j)];
                double uip1 = u[IDX(i+1,j)];
                double uim1 = u[IDX(i-1,j)];
                double ujp1 = u[IDX(i,j+1)];
                double ujm1 = u[IDX(i,j-1)];

                double dudx2 = (uip1 - 2.0*uij + uim1) / (dx*dx);
                double dudy2 = (ujp1 - 2.0*uij + ujm1) / (dy*dy);

                u_new[IDX(i,j)] = uij + dt * alpha * (dudx2 + dudy2);
            }
        }

        // Copy boundaries from u (Dirichlet = 0)
        apply_dirichlet_boundaries(u_new);

        // Swap pointers u <-> u_new
        double *tmp = u;
        u = u_new;
        u_new = tmp;

        // Print simple progress
        if ((n+1) % 100 == 0) {
            printf("Step %d / %d\n", n+1, NSTEPS);
        }
    }

    // Save final field to a CSV file for later visualization
    FILE *f = fopen("u_final.csv", "w");
    if (!f) {
        perror("fopen");
    } else {
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                fprintf(f, "%g", u[IDX(i,j)]);
                if (j < NY - 1) fprintf(f, ",");
            }
            fprintf(f, "\n");
        }
        fclose(f);
        printf("Saved final field to u_final.csv\n");
    }

    free(u);
    free(u_new);

    return 0;
}
