# MPIAsyncPools.jl

This package is for implementing straggler-resilient distributed algorithms and methods, e.g., [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), in systems composed of a coordinator process and multiple worker processes. Stragglers are workers that experience bursts of high latency, e.g., due to network congestion. These workers may, e.g., significantly reduce the rate of convergence of distributed optimization methods, unless the implementation is resilient against stragglers.

To initiate communication, one creates an `MPIAsyncPool` at the coordinator, which handles connecting to the workers via `MPI.Irecv!` and `MPI.Isend`. For example:

```julia
pool = MPIAsyncPool([1, 2, 4]) # pool consisting of nodes with MPI ranks 1, 2, 4
pool = MPIAsyncPool(4)         # pool consisting of nodes with MPI ranks 1, 2, 3, 4
```

Next, the `asyncmap!` function is used to broadcast data, e.g., an iterate (in the case of gradient descent), to the workers and to collect responses. `asyncmap!` returns once results have been received from the `nwait` fastest workers. Alternatively, one can define a custom condition, e.g., to always wait for worker 1.

The workers communicate using regular `MPI.Irecv!` and `MPI.Isend`; see the [MPI.jl](https://github.com/JuliaParallel/MPI.jl) documentation, and the [examples directory](./examples/) for a complete example of how to use `asyncmap!`.


The docstring of `asyncmap!` is:
> `asyncmap!(pool::MPIAsyncPool, sendbuf::AbstractArray, recvbuf::AbstractArray, isendbuf::AbstractArray, irecvbuf::AbstractArray, comm::MPI.Comm; nwait::Union{<:Integer,Function}=pool.nwait, epoch::Integer=pool.epoch+1, tag::Integer=0)`
>
> Send the data in `sendbuf` asynchronously (via `MPI.Isend`) to all workers and wait for some of them to respond (via a corresponding `MPI.Isend`). If `nwait` is an integer, this function returns  when at least `nwait` workers have responded. On the other hand, if `nwait` is a function, this function returns once `nwait(epoch, repochs)` evaluates to `true`. The second argument `repochs` is a vector of length equal to the number of workers in the pool, where the `i`-th element is the `epoch` that transmission to the corresponding worker was initiated at.
>
> Returns the `repochs` vector. Similarly to the `MPI.Gather!` function, the elements of `recvbuf` are partitioned into a number of chunks equal to the number of workers (hence, the length of `recvbuf` must be divisible by the number of workers) and the data recieved from the `j`-th worker is stored in the `j`-th partition.
>
> The buffers `isendbuf` and `irecvbuf` are for internal use by this function only and should never be changed or accessed outside of it. The length of `isendbuf` must be equal to the length of `sendbuf` multiplied by the number of workers, and `irecvbuf` must have length equal to that of `recvbuf`.