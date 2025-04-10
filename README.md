# Lab 3: TeraSort -- Final Project

In this lab, your task is to implement a distributed sorting algorithm along the lines discussed in the lesson videos.  Your design should be based on the assumption that the data will *not* fit in the physical memory of any one computer in your cluster but that it *will* fit in the union of all the physical memories.  This assumption makes your task slightly different from classical 
[distributed disk sorting benchmarks](http://sortbenchmark.org/), but many of the same principles apply.

## Getting Started
Begin by obtaining the starter code from the github repository.

<pre><code>
    git clone https://github.gatech.edu/omscse6220/lab3sp25.git
</code></pre>

Note that this is the [GT github server](https://github.gatech.edu), so you will need to authenticate with your GT credentials.

Optionally, you may choose use a git hosting service for your code.  As always, please **do not make your repository public** on Github or another git hosting service.  Anyone will be able to see it.  If you feel the need to use a hosting service for your project, please keep your repository private.  Your GT account allows you to create free private repositories on [the GT github server](https://github.gatech.edu).

## Programming
The data type you will sorting is similar to the one used in [Hadoop's Terasort benchmark](https://hadoop.apache.org/docs/current/api/org/apache/hadoop/examples/terasort/package-summary.html#package_description).  It consists of a 10-byte key and an 88-byte value.  Procedures for input/output and comparison of these records are provided for you in the **terarec.h** and **terarec.c** files in the *utils* directory.  Your only task is to complete the implementation of the terasort function, which has the signature

<pre><code>
	void terasort(terarec_t *local_data, int local_len, terarec_t **sorted_data, int* sorted_counts, long* sorted_displs);
</code></pre>

Here local_data is the data initially stored in memory at the calling node and local_len is the number of elements in this array.  The remaining parameters are for return values.  The arrays sorted_count and sorted_displs indicate the length and displacement of each node's sorted segment within the whole.  Thus, the data returned to node rank represents the sorted sequence starting at sorted_displs[rank] and has length sorted_count[rank].  The data itself is returned in \*sorted_data, memory which the terasort function itself allocates.

This function can be found in **terasort.c** in the *student* directory. An example of the function's usage can be found in the **terasort_main.c** in *utils*.

## Testing Your Code and Measuring Performance
Performance tests will be a large part of your evaluation. You are encouraged share your ideas and results on the discussion forum.

An example how to check the correctness of your code locally would be:
<pre><code>
$ make clean
$ make teragen teravalidate
$ make terasort
$ mpirun -np 1 ./teragen -c 40000
$ mpirun -np 4 ./terasort
$ mpirun -np 4 ./teravalidate
</code></pre>

We've also included the *terametrics* utility to assist with performance testing:
<pre><code>
$ make clean
$ make teragen terametrics
$ mpirun -np 1 ./teragen -c 160000
$ mpirun -np 16 ./terametrics -c 10
</code></pre>

The following torque jobs are provided for your testing on the cluster:

* example_single.sub (single node - 1x16)
* example_multi.sub (multiple node - 4x16)
* example_dedicated.sub (dedicated multiple node - 4x16)
* pace_interactive.sh (you will need to source *pace_env.sh* like prior labs)

You can get the usage of every generated executable file (e.g. teragen, terasort, teravalidate, terametrics) by running `./tera[xx] -h`.

## Submission

Lab submissions for this class will be through Canvas under the appropriate assignment.

The file you need to submit for this lab is:

```sh
   terasort.c
```
This file should exist in the **student** directory of your lab folder.

Do **not** submit or modify any other files.

If you include any additional headers or libraries, please make sure they compile as-is on the VM and PACE.

## Grading

Your grade on this lab will consist of the following:

* Successful compilation
* Provided test cases
* Additonal test cases
* Performance score

The point totals of each are listed in the forum announcement post.
