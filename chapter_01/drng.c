// compile: gcc drng.c -o drng -mrdrnd

#include <stdio.h>
#include <stdlib.h>

int random_hw(unsigned long long *rnd) {
    int i;

    for (i=0; i<10; i++)
        if (__builtin_ia32_rdrand64_step(rnd))
            return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    int i;
    FILE *f;
    unsigned long long r;

    f = fopen(argv[2],"w");

    for(i=0; i<atoi(argv[1]); i++) {
        if (random_hw(&r)) {
            fwrite((void *)&r, 8, 1, f);
        } else {
            return 0;
        }
    }

    fclose(f);
    return 1;
}


