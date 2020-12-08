#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <signal.h>
#include <omp.h>
#include <mpi-ext.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N   (256+2)


int fault_proc = 1; // processor num who would be killed
int it_of_fault_num = 64; // iteration num where process would be killed
long int check_point = 50; // checkpoint (can set in command line)


double maxeps = 0.1e-7;
int itmax = 100;
int it = 1;
int was_not_faults = 1;
int i,j,k;
int ll, shift;


double reductEps;
double reductSum;
double eps;
double (*A)[N*N];


MPI_Comm my_comm = MPI_COMM_WORLD; // current communicator
MPI_Request req[2]; // for Isend and Irecv
int old_rank; // rank to define who would be killed
int myrank, ranksize; // current rank and number of processes in communicator
int nrow; // number of rows for current process
int delta; // total number of lines from previous processes (for init)
MPI_Status status[2]; // check status after Waitall


void init(); // start initialization of 3d matrix
void compute_delta_and_nrow(); // compute delta and nrow for curr process
void restore_data(); // restore data after break processes
void save_data(); // save data on checkpoint
void verify(); // compute sum
void relax(); // perform calculations
static void verbose_errhandler(MPI_Comm* comm, int* err, ...); // errors hand



int main(int argc, char **argv) {
    MPI_Errhandler errh;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(my_comm, &myrank);
    MPI_Comm_size(my_comm, &ranksize);
    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    MPI_Comm_set_errhandler(my_comm, errh);
    MPI_Barrier(my_comm);
    old_rank = myrank;

    if (argc >= 2) check_point = strtol(argv[1], NULL, 10);
    if (argc >= 3) fault_proc = strtol(argv[2], NULL, 10);
    if (argc >= 4) it_of_fault_num = strtol(argv[3], NULL, 10);

    compute_delta_and_nrow();

    A = calloc((nrow + 2) * N * N, sizeof(double));
    double start = MPI_Wtime();
    init();
    save_data();

    for(it = 1; it <= itmax; it++) {
        MPI_Barrier(my_comm);

        // checkpoint
        if (it % check_point == 0 && was_not_faults) save_data();

        was_not_faults = 1;

        eps = 0.;
        if (myrank != ranksize - 1)
            MPI_Irecv(&A[nrow+1][0], N*N, MPI_DOUBLE, myrank + 1, 1216, 
                      my_comm, &req[0]);

        if (myrank != 0)
             MPI_Isend(&A[1][0], N*N, MPI_DOUBLE, myrank - 1, 1216, 
                       my_comm, &req[1]);

        ll = 2; shift = 0;
        if (myrank == 0) ll = 1;

        if (myrank == ranksize - 1) ll = 1, shift = 1;

        if (ranksize > 1) MPI_Waitall(ll, &req[shift], &status[0]);

        relax();

        if (myrank == 0) printf( "it=%4i   eps=%f\n", it, reductEps);

        if (reductEps < maxeps) break;
    }

    verify();
    if (myrank == 0) {
        printf("  S = %f\n",reductSum);
        remove("store_data");
    }

    double time = MPI_Wtime() - start;
    printf(" Process number = %d, time =  %f \n", myrank, time);
    free(A);
    MPI_Finalize();
    return 0;
}


void init() {
    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
                A[i][j*N + k] = ( 4. + i + delta + j + k);
}


void compute_delta_and_nrow() {
    int startrow = ((myrank * (N-2)) / ranksize) + 1;
    int lastrow = ((myrank + 1) * (N-2)) / ranksize;
    nrow = lastrow - startrow + 1;
    delta = 0;

    // delta computing
    for (int i = 0; i < myrank; ++i) {
        startrow = ((i * (N-2)) / ranksize) + 1;
        lastrow = ((i + 1) * (N-2)) / ranksize;
        delta += lastrow - startrow + 1;
    }
}


void restore_data() {
    free(A);
    compute_delta_and_nrow();
    FILE* f = fopen("store_data", "rb");
    A = calloc((nrow + 2) * N * N, sizeof(double));

    fread(&it, sizeof(int), 1, f);
    fseek(f, delta * N * N * sizeof(double), SEEK_CUR);
    fread(&A[1], sizeof(double), nrow * N * N, f);
    fclose(f);
}


static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...) {
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;

    MPI_Error_class(err, &eclass);
    if (MPIX_ERR_PROC_FAILED != eclass) {
        printf("ABORT!! %d / %d", myrank, ranksize); 
        MPI_Abort(comm, err);
    }

    MPI_Comm_rank(my_comm, &rank);
    MPI_Comm_size(my_comm, &size);

    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    MPI_Group_size(group_f, &nf);

    MPI_Error_string(err, errstr, &len);
    printf("Rank %d / %d:  Notified of error %s. %d found dead: ( ", 
            rank, size, errstr, nf);

    ranks_gf = (int*)malloc(nf * sizeof(int));
    ranks_gc = (int*)malloc(nf * sizeof(int));

    MPI_Comm_group(comm, &group_c);

    for (i = 0; i < nf; ++i) ranks_gf[i] = i;

    MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);

    for (i = 0; i < nf; ++i) { printf("%d ", ranks_gc[i]); }

    printf(")\n");

    MPIX_Comm_shrink(comm, &my_comm);
    MPI_Comm_rank(my_comm, &myrank);
    MPI_Comm_size(my_comm, &ranksize);

    restore_data();

    was_not_faults = 0;
    free(ranks_gc); free(ranks_gf);
}


void save_data() {
    int ok = 1;
    if (myrank == 0) {
        FILE* f = fopen("store_data", "wb");
        int new_it = it - 1;
        fwrite(&new_it, sizeof(int), 1, f);
        fwrite(&A[1], sizeof(double), nrow * N * N, f);
        fclose(f);
        if (ranksize > 1)
            MPI_Send(&ok, 1, MPI_INT, myrank + 1, 1230, my_comm);
    } else {
        MPI_Recv(&ok, 1, MPI_INT, myrank - 1, 1230, my_comm, MPI_STATUS_IGNORE);
        FILE* f = fopen("store_data", "ab");
        fwrite(&A[1], sizeof(double), nrow * N * N, f);
        fclose(f);
        if (myrank != ranksize - 1)
            MPI_Send(&ok, 1, MPI_INT, myrank + 1, 1230, my_comm);
    }
}


void relax() {
    for(j=1; j<=N-2; j++) {
        if (myrank != 0)
            MPI_Recv(&A[0][j*N], N, MPI_DOUBLE, myrank - 1, 1215, 
                     my_comm, &status[0]);

        for(i=1; i<=nrow; i++)
            for(k=1; k<=N-2; k++)
                A[i][j*N + k] = (A[i-1][j*N + k]+A[i+1][j*N + k])*0.5;

        if (myrank != ranksize - 1)
            MPI_Send(&A[nrow][j*N], N, MPI_DOUBLE, myrank + 1, 1215, 
                     my_comm);
    }

    MPI_Barrier(my_comm);

    // process killed himself ///////////
    if (old_rank == fault_proc && it == it_of_fault_num) {
        printf("process %d / %d: bye bye!\n", myrank, ranksize);
        raise(SIGKILL); 
    }
    /////////////////////////////////////

    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
                A[i][j*N + k] = (A[i][(j-1)*N + k]+A[i][(j+1)*N + k])*0.5;

    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++) {
                double e;
                e = A[i][j*N + k];
                A[i][j*N+k] = (A[i][j*N + k-1]+A[i][j*N + k+1])*0.5;
                eps = Max(eps, fabs(e - A[i][j*N + k]));
            }
    MPI_Barrier(my_comm);
    MPI_Reduce(&eps, &reductEps, 1, MPI_DOUBLE, MPI_MAX, 0, my_comm);
    MPI_Bcast(&reductEps, 1, MPI_DOUBLE, 0, my_comm);
}


void verify() {
    double s = 0.;
    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
                s = s + (A[i][j*N + k]*(i + delta + 1)*(j+1)*(k+1)/(N*N)) / N;

    MPI_Reduce(&s, &reductSum, 1, MPI_DOUBLE, MPI_SUM, 0, my_comm);
}