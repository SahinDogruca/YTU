#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h> // Include this header for SIMD intrinsics

#define ARRAY_SIZE 1000000

// Function to add two arrays using SIMD intrinsics
void addArrays(float *a, float *b, float *result, int size)
{
    int i;
    __m128 vec_a, vec_b, vec_result;

    // Process the arrays in chunks of 4 elements (128 bits)
    for (i = 0; i < size; i += 4)
    {
        // Load 4 elements from array 'a' into a SIMD register
        vec_a = _mm_load_ps(&a[i]);

        // Load 4 elements from array 'b' into a SIMD register
        vec_b = _mm_load_ps(&b[i]);

        // Perform SIMD addition
        vec_result = _mm_add_ps(vec_a, vec_b);

        // Store the result back into memory
        _mm_store_ps(&result[i], vec_result);
    }
}

int main()
{
    float *a, *b, *result;
    int i;

    // Allocate memory for arrays
    a = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    b = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    result = (float *)malloc(sizeof(float) * ARRAY_SIZE);

    // Initialize arrays with random values
    for (i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)(rand() % 100);
        b[i] = (float)(rand() % 100);
    }

    // Perform vectorized addition
    addArrays(a, b, result, ARRAY_SIZE);

    // Display the first few elements of the result array
    for (i = 0; i < 10; i++)
    {
        printf("%.2f + %.2f = %.2f\n", a[i], b[i], result[i]);
    }

    // Free allocated memory
    free(a);
    free(b);
    free(result);

    return 0;
}
