#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {

    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {

        for (int j = 0; j < n - i - 1; j++) {

            if (arr[j] > arr[j + 1]) {

                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort
void parallelBubbleSort(vector<int>& arr) {

    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {

        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {

            if (arr[j] > arr[j + 1]) {

                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge Function
void merge(vector<int>& arr, int left, int mid, int right) {

    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {

        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

// Sequential Merge Sort
void sequentialMergeSort(vector<int>& arr, int left, int right) {

    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    sequentialMergeSort(arr, left, mid);
    sequentialMergeSort(arr, mid + 1, right);

    merge(arr, left, mid, right);
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int left, int right) {

    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        parallelMergeSort(arr, left, mid);

        #pragma omp section
        parallelMergeSort(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

// Print Array
void printArray(vector<int>& arr) {

    for (int num : arr)
        cout << num << " ";

    cout << endl;
}

int main() {

    vector<int> original = {64, 34, 25, 12, 22, 11, 90};

    vector<int> arr1 = original;
    vector<int> arr2 = original;
    vector<int> arr3 = original;
    vector<int> arr4 = original;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();

    sequentialBubbleSort(arr1);

    end = omp_get_wtime();

    cout << "\nSequential Bubble Sort:\n";
    printArray(arr1);

    cout << "Time: " << end - start << " seconds\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();

    parallelBubbleSort(arr2);

    end = omp_get_wtime();

    cout << "\nParallel Bubble Sort:\n";
    printArray(arr2);

    cout << "Time: " << end - start << " seconds\n";

    // Sequential Merge Sort
    start = omp_get_wtime();

    sequentialMergeSort(arr3, 0, arr3.size() - 1);

    end = omp_get_wtime();

    cout << "\nSequential Merge Sort:\n";
    printArray(arr3);

    cout << "Time: " << end - start << " seconds\n";

    // Parallel Merge Sort
    start = omp_get_wtime();

    parallelMergeSort(arr4, 0, arr4.size() - 1);

    end = omp_get_wtime();

    cout << "\nParallel Merge Sort:\n";
    printArray(arr4);

    cout << "Time: " << end - start << " seconds\n";

    return 0;
}

/*

(Parallel Bubble Sort and Merge Sort using OpenMP)

---

## 1. What is sorting?

Sorting is the process of arranging data in a specific order such as ascending or descending order.

Example:

```text id="jlwm2p"
5 2 8 1 → 1 2 5 8
```

---

## 2. What is Bubble Sort?

Bubble Sort repeatedly compares adjacent elements and swaps them if they are in the wrong order.

Largest elements move toward the end after every pass.

---

## 3. What is the time complexity of Bubble Sort?

Worst-case time complexity:

```text id="jlwm2q"
O(n²)
```

because nested loops are used.

---

## 4. What is Merge Sort?

Merge Sort is a divide-and-conquer algorithm that:

* divides array into halves
* sorts recursively
* merges sorted parts

---

## 5. What is the time complexity of Merge Sort?

Time complexity:

```text id="jlwm2r"
O(n log n)
```

which is faster than Bubble Sort for large datasets.

---

## 6. Why is Merge Sort faster than Bubble Sort?

Because Merge Sort:

* divides problem into smaller parts
* performs fewer comparisons
* has lower time complexity

---

## 7. What is parallel computing?

Parallel computing executes multiple tasks simultaneously using multiple processors or threads.

It improves execution speed and performance.

---

## 8. What is OpenMP?

OpenMP is an API used for parallel programming in C/C++ and supports multithreading.

It uses compiler directives like:

```cpp id="jlwm2s"
#pragma omp parallel
```

---

## 9. What is multithreading?

Multithreading means executing multiple threads simultaneously within a program.

Each thread performs part of the task independently.

---

## 10. Why is Merge Sort suitable for parallelization?

Because its left and right halves can be sorted independently at the same time.

This fits parallel execution very well.

---

## 11. What is `#pragma omp parallel for`?

It distributes loop iterations among multiple threads for parallel execution.

Example:

```cpp id="jlwm2t"
#pragma omp parallel for
```

---

## 12. What is `#pragma omp parallel sections`?

It executes different code sections simultaneously using separate threads.

Used in parallel Merge Sort.

---

## 13. Why do we measure execution time?

To compare performance between:

* sequential algorithms
* parallel algorithms

and observe speed improvement.

---

## 14. What is speedup in parallel computing?

Speedup measures performance improvement.

Formula:

```text id="jlwm2u"
Speedup = Sequential Time / Parallel Time
```

---

## 15. What is divide-and-conquer?

A technique where a problem is:

* divided into smaller subproblems
* solved recursively
* combined to form final solution

Used in Merge Sort.

---

## 16. Which sorting algorithm is better for large datasets?

Merge Sort is better because:

* faster execution
* lower complexity
* efficient parallelization

---

## 17. What are advantages of parallel sorting?

* Faster execution
* Better CPU utilization
* Handles large data efficiently
* Reduces processing time

---

## 18. What are disadvantages of Bubble Sort?

* Slow for large datasets
* High time complexity
* Inefficient compared to advanced algorithms

---

## 19. What is recursion?

A function calling itself repeatedly until a base condition is reached.

Used in Merge Sort.

---

## 20. Difference between sequential and parallel algorithms?

| Sequential                  | Parallel                               |
| --------------------------- | -------------------------------------- |
| Executes one task at a time | Executes multiple tasks simultaneously |
| Slower                      | Faster                                 |
| Uses single processor       | Uses multiple threads/processors       |
*/