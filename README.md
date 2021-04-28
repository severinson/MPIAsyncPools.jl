# MPIStragglers.jl

This package is for assigning tasks to workers in distributed systems composed of a coordinator process and multiple worker processes. Each worker communicates with the coordinator using MPI, and the results computed by the workers are collected by the coordinator in an asynchronous manner. This is particularly useful for high-performance computing applications in clusters with straggling workers, i.e., where the latency may differ significantly between workers and iterations.

First, one creates a worker pool from the MPI ranks of the worker processes. For example:

```julia
pool = MPIAsyncPool([1, 2, 4]) # pool consisting of nodes with MPI ranks 1, 2, 4
pool = MPIAsyncPool(4)         # pool consisting of nodes with MPI ranks 1, 2, 3, 4
```

Next, the `asyncmap!` function is used to assign tasks to workers. This function returns once results have been received from the `nwait` fastest workers. Alternatively, one can define a custom condition, e.g., to always wait for worker 1. The docstring of `asyncmap!` is:

> `asyncmap!(pool::MPIAsyncPool, sendbuf::AbstractArray, recvbuf::AbstractArray, isendbuf::AbstractArray, irecvbuf::AbstractArray, comm::MPI.Comm; nwait::Union{<:Integer,Function}=pool.nwait, epoch::Integer=pool.epoch+1, tag::Integer=0)`
>
> Send the data in `sendbuf` asynchronously (via `MPI.Isend`) to all workers and wait for some of them to respond (via a corresponding `MPI.Isend`). If `nwait` is an integer, this function returns  when at least `nwait` workers have responded. On the other hand, if `nwait` is a function, this function returns once `nwait(epoch, repochs)` evaluates to `true`. The second argument `repochs` is a vector of length equal to the number of workers in the pool, where the `i`-th element is the `epoch` that transmission to the corresponding worker was initiated at.
>
> Returns the `repochs` vector. Similarly to the `MPI.Gather!` function, the elements of `recvbuf` are partitioned into a number of chunks equal to the number of workers (hence, the length of `recvbuf` must be divisible by the number of workers) and the data recieved from the `j`-th worker is stored in the `j`-th partition.
>
> The buffers `isendbuf` and `irecvbuf` are for internal use by this function only and should never be changed or accessed outside of it. The length of `isendbuf` must be equal to the length of `sendbuf` multiplied by the number of workers, and `irecvbuf` must have length equal to that of `recvbuf`.