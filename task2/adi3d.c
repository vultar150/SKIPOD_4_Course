#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N   (256+2)
double   maxeps = 0.1e-7;
int itmax = 10;
int i,j,k;
int ll, shift;

double reductEps;
double reductSum;
double eps;
double (*A)[N*N];


MPI_Request req[2];
int myrank, ranksize;
int nrow;
int delta;
MPI_Status status[2];


void relax();
void init();
void verify();

int main(int argc, char **argv)
{
    int it;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);

    int startrow = ((myrank * (N-2)) / ranksize) + 1;
    int lastrow = ((myrank + 1) * (N-2)) / ranksize;
    nrow = lastrow - startrow + 1;

    for (int i = 0; i < myrank; ++i) 
    {
        startrow = ((i * (N-2)) / ranksize) + 1;
        lastrow = ((i + 1) * (N-2)) / ranksize;
        delta += lastrow - startrow + 1;
    }

    A = calloc((nrow+2)*N*N , sizeof(double));
    double start = MPI_Wtime();
    init();
    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        if (myrank != ranksize-1)
        {
            MPI_Irecv(&A[nrow+1][0], N*N, MPI_DOUBLE, myrank+1, 1216, MPI_COMM_WORLD, &req[0]);
        }
        if (myrank != 0)
        {
             MPI_Isend(&A[1][0], N*N, MPI_DOUBLE, myrank-1, 1216, MPI_COMM_WORLD, &req[1]);
        }
        ll = 2; shift = 0;
        if (myrank == 0)
        {
            ll = 1;
        }
        if (myrank == ranksize-1)
        {
            ll = 1; shift = 1;
        }
        if (ranksize > 1) {
            MPI_Waitall(ll, &req[shift], &status[0]);
        }
        relax();
        if (myrank == 0)
        {
            printf( "it=%4i   eps=%f\n", it, reductEps);
        }
        if (reductEps < maxeps) break;
    }

    verify();
    if (myrank == 0)
    {
        printf("  S = %f\n",reductSum);
    }
    double time = MPI_Wtime() - start;
    printf(" Process number = %d, time =  %f \n", myrank, time);
    MPI_Finalize();
    return 0;
}


void init()
{
    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j*N + k] = ( 4. + i + delta + j + k);
            }
}


void relax()
{
    for(j=1; j<=N-2; j++)
    {
        if (myrank != 0)
        {
            MPI_Recv(&A[0][j*N], N, MPI_DOUBLE, myrank-1, 1215, MPI_COMM_WORLD, &status[0]);
        }
        for(i=1; i<=nrow; i++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j*N + k] = (A[i-1][j*N + k]+A[i+1][j*N + k])*0.5;
            }
        if (myrank != ranksize-1)
        {
            MPI_Send(&A[nrow][j*N], N, MPI_DOUBLE, myrank+1, 1215, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);


    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                A[i][j*N + k] = (A[i][(j-1)*N + k]+A[i][(j+1)*N + k])*0.5;
            }

    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                double e;
                e = A[i][j*N + k];
                A[i][j*N+k] = (A[i][j*N + k-1]+A[i][j*N + k+1])*0.5;
                eps = Max(eps, fabs(e - A[i][j*N + k]));
            }
    MPI_Reduce(&eps, &reductEps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reductEps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}


void verify()
{
    double s = 0.;
    for(i=1; i<=nrow; i++)
        for(j=1; j<=N-2; j++)
            for(k=1; k<=N-2; k++)
            {
                s = s + (A[i][j*N + k]*(i+delta+1)*(j+1)*(k+1)/(N*N)) /N;
            }
    MPI_Reduce(&s, &reductSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}