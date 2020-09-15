#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#define WEIGHT_PARAMETER 10000
#define TASKS_PER_PROCESS 1000
#define MIN_TASKS_TO_SHARE 10
#define ITERATIONS_COUNT 5

#define MAIN_PROC 0

#define TAG_REQUEST 0
#define TAG_REPLY 1

double global_result = 0.0;
double global_result_sum = 0.0;

int* tasks;
int tasks_remaining;

pthread_mutex_t mutex_tasks;
pthread_mutex_t mutex_tasks_remaining;

pthread_t thread_receiver;

int rank, size;

int Create_threads();
void* Thread_receiver_start(void* args);
void* Thread_worker_start(void* args);
void Set_tasks_weight(int *tasks, int count, int iterCounter);
void Execute_tasks(const int *tasks, int* tasks_executed);

int main(int argc, char **argv) {
	int required = MPI_THREAD_MULTIPLE;
	int provided;

	MPI_Init_thread(&argc, &argv, required, &provided);

	if (provided != required) {
		fprintf(stderr, "MPI_Init_thread(): provided level != required level\n");
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	pthread_mutex_init(&mutex_tasks, NULL);
	pthread_mutex_init(&mutex_tasks_remaining, NULL);

	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();
	int return_value = Create_threads();
	double elapsed_time = MPI_Wtime() - start_time;
	
	pthread_mutex_destroy(&mutex_tasks);
	pthread_mutex_destroy(&mutex_tasks_remaining);

	if (return_value != EXIT_SUCCESS) {
		fprintf(stderr, "Process #%d failed, its termination...\n", rank + 1);
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	if (rank == MAIN_PROC) {
		printf("-------------------------------------------------\n");
		printf("Total elapsed time:\t%lf sec.\n", elapsed_time);
		printf("Global result sum:\t%lf\n", global_result_sum);
		printf("-------------------------------------------------\n");
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}

int Create_threads() {
	pthread_attr_t attr;

	if (pthread_attr_init(&attr)) {
		fprintf(stderr, "Could not initialize attributes\n");
		return EXIT_FAILURE;
	}

	if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE)) {
		fprintf(stderr, "Could not set attributes\n");
		return EXIT_FAILURE;
	}

	if (pthread_create(&thread_receiver, &attr, Thread_receiver_start, NULL)) {
		fprintf(stderr, "Could not create Thread 'Receiver'\n");
		return EXIT_FAILURE;
	}

	pthread_attr_destroy(&attr);

	Thread_worker_start(NULL);

	if (pthread_join(thread_receiver, NULL)) {
		fprintf(stderr, "Could not join Thread 'Receiver'\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void* Thread_receiver_start(void* args) {
	int tasks_to_share;
	int rank_requested;

	while (1) {
		MPI_Recv(
				/* buf 		= */ &rank_requested,
				/* count 	= */ 1,
				/* datatype = */ MPI_INT,
				/* source 	= */ MPI_ANY_SOURCE,
				/* tag 		= */ TAG_REQUEST,
				/* comm 	= */ MPI_COMM_WORLD,
				/* status 	= */ MPI_STATUS_IGNORE);

		if (rank_requested == rank) break;
		
		pthread_mutex_lock(&mutex_tasks_remaining);
		if (tasks_remaining >= MIN_TASKS_TO_SHARE) {
			tasks_to_share = tasks_remaining / 2;
			tasks_remaining -= tasks_to_share;

			MPI_Send(
				/* buf 		= */ &tasks_to_share,
				/* count 	= */ 1,
				/* datatype = */ MPI_INT,
				/* dest 	= */ rank_requested,
				/* tag 		= */ TAG_REPLY,
				/* comm 	= */ MPI_COMM_WORLD);

			pthread_mutex_lock(&mutex_tasks);
			MPI_Send(
				/* buf 		= */ tasks + tasks_remaining - 1,
				/* count 	= */ tasks_to_share,
				/* datatype = */ MPI_INT,
				/* dest 	= */ rank_requested,
				/* tag 		= */ TAG_REPLY,
				/* comm 	= */ MPI_COMM_WORLD);
			pthread_mutex_unlock(&mutex_tasks);
		} else {
			tasks_to_share = 0;

			MPI_Send(
				/* buf 		= */ &tasks_to_share,
				/* count 	= */ 1,
				/* datatype = */ MPI_INT,
				/* dest 	= */ rank_requested,
				/* tag 		= */ TAG_REPLY,
				/* comm 	= */ MPI_COMM_WORLD);
		}
		pthread_mutex_unlock(&mutex_tasks_remaining);
	}
	return NULL;
}

void* Thread_worker_start(void* args) {
	tasks = (int*)malloc(sizeof(int) * TASKS_PER_PROCESS);

	double start_time, elapsed_time;
	double elapsed_time_min, elapsed_time_max;

	for (int iterCounter = 0; iterCounter < ITERATIONS_COUNT; ++iterCounter) {
		Set_tasks_weight(tasks, TASKS_PER_PROCESS, iterCounter);

		pthread_mutex_lock(&mutex_tasks_remaining);
		tasks_remaining = TASKS_PER_PROCESS;
		pthread_mutex_unlock(&mutex_tasks_remaining);
		int tasks_executed = 0;
		int tasks_received;

		MPI_Barrier(MPI_COMM_WORLD);
		start_time = MPI_Wtime();

		Execute_tasks(tasks, &tasks_executed);

		for (int rank_iter = 0; rank_iter < size; ++rank_iter) {
			if (rank_iter == rank) continue;

			// Send message "I, Process #rank, have finished my work and can execute some extra tasks!"
			MPI_Send(
				/* buf 		= */ &rank,
				/* count 	= */ 1,
				/* datatype = */ MPI_INT,
				/* dest 	= */ rank_iter,
				/* tag 		= */ TAG_REQUEST,
				/* comm 	= */ MPI_COMM_WORLD);

			// Receive number of extra tasks
			MPI_Recv(
				/* buf 		= */ &tasks_received,
				/* count 	= */ 1,
				/* datatype = */ MPI_INT,
				/* source 	= */ rank_iter,
				/* tag 		= */ TAG_REPLY,
				/* comm 	= */ MPI_COMM_WORLD,
				/* status 	= */ MPI_STATUS_IGNORE);

			if (tasks_received > 0) {
				// Receive tasks
				MPI_Recv(
					/* buf 		= */ tasks,
					/* count 	= */ tasks_received,
					/* datatype = */ MPI_INT,
					/* source 	= */ rank_iter,
					/* tag 		= */ TAG_REPLY,
					/* comm 	= */ MPI_COMM_WORLD,
					/* status 	= */ MPI_STATUS_IGNORE);

				pthread_mutex_lock(&mutex_tasks_remaining);
				tasks_remaining = tasks_received;
				pthread_mutex_unlock(&mutex_tasks_remaining);

				Execute_tasks(tasks, &tasks_executed);
			}
		}

		elapsed_time = MPI_Wtime() - start_time;

		MPI_Allreduce(
			/* sendbuf 	= */ &elapsed_time,
			/* recvbuf 	= */ &elapsed_time_min,
			/* count 	= */ 1,
			/* datatype = */ MPI_DOUBLE,
			/* op 		= */ MPI_MIN,
			/* comm 	= */ MPI_COMM_WORLD);

		MPI_Allreduce(
			/* sendbuf 	= */ &elapsed_time,
			/* recvbuf 	= */ &elapsed_time_max,
			/* count 	= */ 1,
			/* datatype = */ MPI_DOUBLE,
			/* op 		= */ MPI_MAX,
			/* comm 	= */ MPI_COMM_WORLD);

		// Print info about this iteration
		if (rank == MAIN_PROC) {
			printf("-------------------------------------------------\n");
			printf("Iteration #%d\n", iterCounter + 1);
			printf("Disbalance time: %lf sec.\n", elapsed_time_max - elapsed_time_min);
			printf("Disbalance perc: %lf %%\n", (elapsed_time_max - elapsed_time_min) / elapsed_time_max * 100);
		}
		for (int rank_iter = 0; rank_iter < size; rank_iter++) {
			if (rank == rank_iter) {
				printf("\tProcess #%d\n", rank + 1);
				printf("\t\texecuted tasks:\t%d\n", tasks_executed);
				printf("\t\tglobal result:\t%lf\n", global_result);
				printf("\t\titeration time:\t%lf sec.\n", elapsed_time);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	// Terminate Thread 'Receiver'
	MPI_Send(
				/* buf 		= */ &rank,
				/* count 	= */ 1,
				/* datatype = */ MPI_INT,
				/* dest 	= */ rank,
				/* tag 		= */ 0,
				/* comm 	= */ MPI_COMM_WORLD);

	MPI_Allreduce(
		/* sendbuf 	= */ &global_result,
		/* recvbuf 	= */ &global_result_sum,
		/* count 	= */ 1,
		/* datatype = */ MPI_DOUBLE,
		/* op 		= */ MPI_SUM,
		/* comm 	= */ MPI_COMM_WORLD);


	free(tasks);
	return NULL;
}

void Set_tasks_weight(int *tasks, int count, int iterCounter) {
	pthread_mutex_lock(&mutex_tasks);
	for (int i = 0; i < count; ++i) {
		tasks[i] = abs(50 - i % 100) * abs(rank - (iterCounter % size)) * WEIGHT_PARAMETER;
	}
	pthread_mutex_unlock(&mutex_tasks);
}

void Execute_tasks(const int *tasks, int* tasks_executed) {
	pthread_mutex_lock(&mutex_tasks_remaining);
	for (int i = 0; tasks_remaining; ++i, --tasks_remaining) {
		pthread_mutex_unlock(&mutex_tasks_remaining);

		pthread_mutex_lock(&mutex_tasks);
		int task_weight = tasks[i];
		pthread_mutex_unlock(&mutex_tasks);

		for (int j = 0; j < task_weight; ++j) {
			global_result += sin(j);
		}

		++(*tasks_executed);
		pthread_mutex_lock(&mutex_tasks_remaining);
	}
	pthread_mutex_unlock(&mutex_tasks_remaining);
}