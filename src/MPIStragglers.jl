module MPIStragglers

using MPI

export AsyncPool, asyncmap!

"""

    AsyncPool(ranks::Vector{<:Integer}; epoch0::Integer=0, nwait=length(ranks))

Used to manage a pool of straggling workers that communicate over MPI. Stores the state required 
for `asyncmap!`. `nwait` is the default number of workers to wait for when calling `asyncmap!` 
and `epoch0` is the epoch of the first iteration.

```julia
AsyncPool(n)            # create a pool composed of n workers with ranks 1:n
AsyncPool([1, 4, 5])    # create a pool composed of 3 workers with ranks [1, 4, 5]
```
"""
struct AsyncPool
    ranks::Vector{Int}
    sreqs::Vector{MPI.Request}
    rreqs::Vector{MPI.Request}
    sepochs::Vector{Int}
    repochs::Vector{Int}
    active::Vector{Bool}
    nwait::Integer # default number of workers to wait for
    function AsyncPool(ranks::Vector{<:Integer}; epoch0::Integer=0, nwait=length(ranks))
        n = length(ranks)
        new(copy(ranks),
            Vector{MPI.Request}(undef, n), Vector{MPI.Request}(undef, n),
            Vector{Int}(undef, n), fill(epoch0, n),
            zeros(Bool, n),
            nwait)
    end
end

AsyncPool(n::Integer, args...; kwargs...) = AsyncPool(collect(1:n), args...; kwargs...)

"""
    asyncmap!(sendbuf, recvbuf, isendbuf, irecvbuf, nwait::Union{<:Integer,Function}, epoch::Integer, pool::AsyncPool, comm::MPI.Comm)

Send the data in `sendbuf` asynchronously (via `MPI.Isend`) to all workers and wait for some of
them to respond (via a corresponding `MPI.Isend`). If `nwait` is an integer, this function returns 
when at least `nwait` workers have responded. On the other hand, if `nwait` is a function, this 
function returns once `nwait(epoch, repochs)` evaluates to `true`. The second argument `repochs` is 
a vector of length equal to the number of workers in the pool, where the `i`-th element is the 
`epoch` that transmission to the corresponding worker was initiated at. For iterative algorithms, 
`epoch` should be the iteration that the data in `sendbuf` corresponds to.

Returns the `repochs` vector. Similarly to the `MPI.Gather!` function, the elements of `recvbuf` 
are partitioned into a number of chunks equal to the number of workers (hence, the length of 
`recvbuf` must be divisible by the number of workers) and the data recieved from the `j`-th worker 
is stored in the `j`-th partition.

The buffers `isendbuf` and `irecvbuf` are for internal use by this function only and should never 
be changed or accessed outside of it. The length of `isendbuf` must be equal to the length of 
`sendbuf` multiplied by the number of workers, and `irecvbuf` must have length equal to that of 
`recvbuf`.
"""
function asyncmap!(sendbuf::AbstractArray, recvbuf::AbstractArray, isendbuf::AbstractArray, irecvbuf::AbstractArray, nwait::Union{<:Integer,Function}, epoch::Integer, pool::AsyncPool, comm::MPI.Comm; tag::Integer=0)
    comm_size = length(pool.ranks)
    if typeof(nwait) <: Integer
        0 <= nwait <= comm_size || throw(ArgumentError("nwait must be in the range [0, length(pool.ranks)], but is $nwait"))
    end
    isbitstype(eltype(sendbuf)) || throw(ArgumentError("The eltype of sendbuf must be isbits, but is $(eltype(sendbuf))"))
    isbitstype(eltype(recvbuf)) || throw(ArgumentError("The eltype of sendbuf must be isbits, but is $(eltype(recvbuf))"))
    sizeof(isendbuf) == comm_size*sizeof(sendbuf) || throw(DimensionMismatch("sendbuf is of size $(sizeof(sendbuf)) bytes, but isendbuf is of size $(sizeof(isendbuf)) bytes when $(comm_size*sizeof(sendbuf)) bytes are needed"))
    sizeof(recvbuf) == sizeof(irecvbuf) || throw(DimensionMismatch("recvbuf is of size $(sizeof(recvbuf)) bytes, but irecvbuf is of size $(sizeof(irecvbuf)) bytes"))
    mod(length(recvbuf), comm_size) == 0 || throw(DimensionMismatch("The length of recvbuf and irecvbuf must be a multiple of the number of workers"))

    # send/receive buffers for each worker
    sl = length(sendbuf)
    rl = div(length(irecvbuf), comm_size)
    isendbufs = [view(isendbuf, (i-1)*sl+1:i*sl) for i in 1:comm_size]
    irecvbufs = [view(irecvbuf, (i-1)*rl+1:i*rl) for i in 1:comm_size]
    recvbufs = [view(recvbuf, (i-1)*rl+1:i*rl) for i in 1:comm_size]

    # start nonblocking transmission of sendbuf to all inactive workers
    # all workers are active after this loop
    for i in 1:length(pool.ranks)
        
        # do nothing with active workers
        if pool.active[i]
            continue
        end

        # worker is inactive
        pool.active[i] = true
        rank = pool.ranks[i]

        # copy send data to a worker-specific buffer to ensure the data isn't changed while sending
        isendbufs[i] .= view(sendbuf, :)

        # remember at what epoch data was sent
        pool.sepochs[i] = epoch

        # non-blocking send and receive
        pool.sreqs[i] = MPI.Isend(isendbufs[i], rank, tag, comm)
        pool.rreqs[i] = MPI.Irecv!(irecvbufs[i], rank, tag, comm)
    end

    # continually send and receive until,
    # 1) if nwait is an integer, we've received nwait results from this epoch, or
    # 2) if nwait is a function, when nwait(epoch, repochs) evaluates to true.
    # at least k workers are inactive after this loop
    nrecv = 0
    while true

        if typeof(nwait) <: Integer
            if nrecv >= nwait
                break            
            end
        elseif typeof(nwait) <: Function 
            if nwait(epoch, pool.repochs)::Bool
                break
            end
        else
            error("nwait must be either an Integer or a Function, but is a $(typeof(nwait))")
        end

        indices, _ = MPI.Waitsome!(pool.rreqs)
        for i in indices
            rank = pool.ranks[i]

            # store the received data and set the epoch
            recvbufs[i] .= irecvbufs[i]
            pool.repochs[i] = pool.sepochs[i]

            # cleanup the corresponding send request; should return immediately
            MPI.Wait!(pool.sreqs[i])

            # only receives initiated this epoch count towards completion
            if pool.repochs[i] == epoch
                nrecv += 1
                pool.active[i] = false
            else
                isendbufs[i] .= view(sendbuf, :)
                pool.sepochs[i] = epoch
                pool.sreqs[i] = MPI.Isend(isendbufs[i], rank, tag, comm)
                pool.rreqs[i] = MPI.Irecv!(irecvbufs[i], rank, tag, comm)
            end
        end
    end

    return pool.repochs
end

asyncmap!(sendbuf::AbstractArray, recvbuf::AbstractArray, isendbuf::AbstractArray, irecvbuf::AbstractArray, epoch::Integer, pool::AsyncPool, comm::MPI.Comm; tag::Integer=0) = asyncmap!(sendbuf, recvbuf, isendbuf, irecvbuf, pool.nwait, epoch, pool, comm; tag)

end