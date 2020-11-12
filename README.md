# MPIStragglers

This function exports a function `kmap!` for distributing work over straggling workers.

> `kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, k::Integer, epoch::Integer, pool::StragglerPool, comm::MPI.Comm)`
>
>Send the data in `sendbuf` asynchronously (via `MPI.Isend`) to all workers and wait for the fastest `k` workers to respond (via a corresponding `MPI.Isend`). For iterative algorithms, `epoch` should be the iteration that the data in `sendbuf` corresponds to.
>
>Returns a vector of length equal to the number of workers in the pool, where the `i`-th element is the `epoch` that transmission to that worker was initiated at. Similarly to the `MPI.Gather!` function, the elements of `recvbuf` are partitioned into a number of chunks equal to the number of workers (hence, the length of `recvbuf` must be divisible by the number of workers) and the data recieved from the
`j`-th worker is stored in the `j`-th partition.
>
>The buffers `isendbuf` and `irecvbuf` are for internal use by this function only and should never be changed or accessed outside of it. The length of `isendbuf` must be equal to the length of `sendbuf` multiplied by the number of workers, and `irecvbuf` must have length equal to that of `recvbuf`.

It takes a `StragglerPool` as one of its arguments, which is responsible for tracking the state of the workers. Its constructor has the following signature:

> `StragglerPool(ranks::Vector{<:Integer}, epoch0::Integer=0)`

## Examples

```julia
pool = StragglerPool([1, 2, 3]) # pool consisting of nodes with MPI ranks 1, 2, 3
pool = StragglerPool(4) # pool consisting of nodes with MPI ranks 1, 2, 3, 4
```