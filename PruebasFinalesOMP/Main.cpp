/*#include <omp.h>
#include <stdio.h>
int var = 10;
#pragma omp threadprivate(var)
void main() {
    printf("Initial value (%d), Var=%d\n\n", omp_get_thread_num(), var);

#pragma omp parallel  num_threads (2)
    {

        var *= (omp_get_thread_num() + 1);  //Potential race condition.
        printf("Thread (%d), Var=%d\n\n", omp_get_thread_num(), var);
    }
    var += 1000;
#pragma omp parallel copyin(var) num_threads (2)
    {
        printf("Thread (%d), Var=%d\n\n", omp_get_thread_num(), var);
    }
}

*/

#include <stdio.h>

void main() {
    int i;
#pragma omp parallel for num_threads(4) ordered
    for (i = 0; i < 20; i++) {

//#pragma omp ordered
        printf(" %d\n", i);
    }
}

