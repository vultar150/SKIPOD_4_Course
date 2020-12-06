#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stddef.h>
#include <time.h>
#include <mpi.h>


const int MAX_SIZE_NAME = 80;


typedef struct {
    double current_time;
    int rank_of_sender;
    char name_of_critical[MAX_SIZE_NAME];
} Message;


int count = 2;
int array_of_blocklengths[] = { 1, 1, MAX_SIZE_NAME };
MPI_Aint array_of_displacements[] = { offsetof( Message, current_time ),
                                      offsetof( Message, rank_of_sender ),
                                      offsetof( Message, name_of_critical ) };

MPI_Datatype array_of_types[] = { MPI_DOUBLE, MPI_INT, MPI_CHAR };
MPI_Datatype tmp_type, my_mpi_type;
MPI_Aint lb, extent;



int main(int argc, char *argv[]) {
    int error;
    error = MPI_Init(&argc, &argv);

    double start = MPI_Wtime();

    srand(getpid());

    // sleep(rand() % 5 + 1);

    if (error != MPI_SUCCESS) {
        fprintf(stderr, "ERROR: CAN'T MPI INIT\n");
        return 1;
    }

    // my message type registration
    MPI_Type_create_struct( count, array_of_blocklengths, array_of_displacements,
                            array_of_types, &tmp_type );
    MPI_Type_get_extent( tmp_type, &lb, &extent );
    MPI_Type_create_resized( tmp_type, lb, extent, &my_mpi_type );
    MPI_Type_commit( &my_mpi_type );


    int myrank, ranksize;

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

    MPI_Request req[ranksize - 1];
    MPI_Status  status[ranksize - 1];

    Message messages[ranksize]; // for recieve messages
    Message message_for_send;

    message_for_send.rank_of_sender = myrank;
    strcpy(message_for_send.name_of_critical, "critical");
    double timestamp = MPI_Wtime() - start;
    message_for_send.current_time = timestamp;

    char ok_send[3];
    char* ok_recv[3];
    strcpy(ok_send, "OK");

    int j = 0;
    // send request to all processes
    for (int i = (myrank + 1) % ranksize; i != myrank; i = (i + 1) % ranksize)
        MPI_Isend(&message_for_send, 1, my_mpi_type, i, 1215, MPI_COMM_WORLD, &req[j++]);

    j = 0;
    // recieve request from all other processes
    for (int i = (myrank + 1) % ranksize; i != myrank; i = (i + 1) % ranksize)
        MPI_Recv(&messages[i], 1, my_mpi_type, i, 1215, MPI_COMM_WORLD, &status[j++]);

    // waiting until all other processes will recieved message of current process 
    if (ranksize > 1) MPI_Waitall(ranksize - 1, &req[0], &status[0]);
    MPI_Barrier(MPI_COMM_WORLD);

    int num_less = 0;
    for (int i = 0; i < myrank; ++i)
        if (messages[i].current_time <= timestamp)
            MPI_Isend(&ok_send[0], 3, MPI_CHAR, i, 1217, MPI_COMM_WORLD, &req[num_less++]);

    for (int i = myrank + 1; i < ranksize; ++i)
        if (messages[i].current_time < timestamp)
            MPI_Isend(&ok_send[0], 3, MPI_CHAR, i, 1217, MPI_COMM_WORLD, &req[num_less++]);

    j = 0;
    // waiting "OK" from all other processes
    for (int i = (myrank + 1) % ranksize; i != myrank; i = (i + 1) % ranksize)
        MPI_Recv(&ok_recv[0], 3, MPI_CHAR, MPI_ANY_SOURCE, 1217, MPI_COMM_WORLD, &status[j++]);


    /////////////////////////////////////////////
    // start critical section
    // <проверка наличия файла “critical.txt”>;
    FILE *file; 
    file = fopen("critical.txt", "r");
    if (file != NULL) {
        fprintf(stderr, "ERROR: FILE EXIST! PROCESS NUM = %d\n", myrank);
        return 1;
    } else {
        file = fopen("critical.txt", "w");
        // some work with file
        sleep(rand() % 10 + 1);
        fclose(file);
        remove("critical.txt");
    }
    // end critical section
    /////////////////////////////////////////////

    // send "OK" to processes on remaining requests

    for (int i = 0; i < myrank; ++i)
        if (messages[i].current_time > timestamp)
            MPI_Isend(&ok_send[0], 3, MPI_CHAR, i, 1217, MPI_COMM_WORLD, &req[num_less++]);

    for (int i = myrank + 1; i < ranksize; ++i)
        if (messages[i].current_time >= timestamp)
            MPI_Isend(&ok_send[0], 3, MPI_CHAR, i, 1217, MPI_COMM_WORLD, &req[num_less++]);

    // waiting until remaining processes will recieve "OK"
    if (ranksize > 1) MPI_Waitall(ranksize - 1, &req[0], &status[0]);

    // Finalize the MPI environment.
    MPI_Type_free( &my_mpi_type );
    MPI_Finalize();
    return 0;
}


