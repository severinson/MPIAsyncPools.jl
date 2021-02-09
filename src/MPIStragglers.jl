module MPIStragglers

using MPI

export AsyncPool, asyncmap!, waitall!

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
mutable struct AsyncPool
    ranks::Vector{Int}
    sreqs::Vector{MPI.Request}
    rreqs::Vector{MPI.Request}
    sepochs::Vector{Int}
    repochs::Vector{Int}
    active::Vector{Bool}
    nwait::Int # default number of workers to wait for
    epoch::Int # current epoch
    function AsyncPool(ranks::Vector{<:Integer}; epoch0::Integer=0, nwait=length(ranks))
        n = length(ranks)
        new(copy(ranks),
            Vector{MPI.Request}(undef, n), Vector{MPI.Request}(undef, n),
            Vector{Int}(undef, n), fill(epoch0, n),
            zeros(Bool, n),
            nwait, epoch0)
    end
end

AsyncPool(n::Integer, args...; kwargs...) = AsyncPool(collect(1:n), args...; kwargs...)

"""
    asyncmap!(pool::AsyncPool, sendbuf::AbstractArray, recvbuf::AbstractArray, isendbuf::AbstractArray, irecvbuf::AbstractArray, comm::MPI.Comm; nwait::Union{<:Integer,Function}=pool.nwait, epoch::Integer=pool.epoch+1, tag::Integer=0)

Send the data in `sendbuf` asynchronously (via `MPI.Isend`) to all workers and wait for some of
them to respond (via a corresponding `MPI.Isend`). If `nwait` is an integer, this function returns 
when at least `nwait` workers have responded. On the other hand, if `nwait` is a function, this 
function returns once `nwait(epoch, repochs)` evaluates to `true`. The second argument `repochs` is 
a vector of length equal to the number of workers in the pool, where the `i`-th element is the 
`epoch` that transmission to the corresponding worker was initiated at.

Returns the `repochs` vector. Similarly to the `MPI.Gather!` function, the elements of `recvbuf` 
are partitioned into a number of chunks equal to the number of workers (hence, the length of 
`recvbuf` must be divisible by the number of workers) and the data recieved from the `j`-th worker 
is stored in the `j`-th partition.

The buffers `isendbuf` and `irecvbuf` are for internal use by this function only and should never 
be changed or accessed outside of it. The length of `isendbuf` must be equal to the length of 
`sendbuf` multiplied by the number of workers, and `irecvbuf` must have length equal to that of 
`recvbuf`.
"""
function Base.asyncmap!(pool::AsyncPool, sendbuf::AbstractArray, recvbuf::AbstractArray, isendbuf::AbstractArray, irecvbuf::AbstractArray, comm::MPI.Comm; nwait::Union{<:Integer,Function}=pool.nwait, epoch::Integer=pool.epoch+1, tag::Integer=0)
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
    sl = sizeof(sendbuf)
    rl = div(sizeof(irecvbuf), comm_size)
    isendbufs = [view(reinterpret(UInt8, isendbuf), (i-1)*sl+1:i*sl) for i in 1:comm_size]
    irecvbufs = [view(reinterpret(UInt8, irecvbuf), (i-1)*rl+1:i*rl) for i in 1:comm_size]    
    recvbufs = [view(reinterpret(UInt8, recvbuf), (i-1)*rl+1:i*rl) for i in 1:comm_size]

    # each call to asyncmap! is the start of a new epoch
    pool.epoch = epoch

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
        isendbufs[i] .= view(reinterpret(UInt8, sendbuf), :)

        # remember at what epoch data was sent
        pool.sepochs[i] = pool.epoch

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
            if nwait(pool.epoch, pool.repochs)::Bool
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
            if pool.repochs[i] == pool.epoch
                nrecv += 1
                pool.active[i] = false
            else
                isendbufs[i] .= reinterpret(UInt8, view(sendbuf, :))
                pool.sepochs[i] = pool.epoch
                pool.sreqs[i] = MPI.Isend(isendbufs[i], rank, tag, comm)
                pool.rreqs[i] = MPI.Irecv!(irecvbufs[i], rank, tag, comm)
            end
        end
    end

    return pool.repochs
end

"""
    waitall!(pool::AsyncPool, recvbuf::AbstractArray, irecvbuf::AbstractArray)

Wait for all workers to respond. All workers are inactive when this function has returned.
"""
function waitall!(pool::AsyncPool, recvbuf::AbstractArray, irecvbuf::AbstractArray)
    comm_size = length(pool.ranks)        
    isbitstype(eltype(recvbuf)) || throw(ArgumentError("The eltype of sendbuf must be isbits, but is $(eltype(recvbuf))"))
    sizeof(recvbuf) == sizeof(irecvbuf) || throw(DimensionMismatch("recvbuf is of size $(sizeof(recvbuf)) bytes, but irecvbuf is of size $(sizeof(irecvbuf)) bytes"))
    mod(length(recvbuf), comm_size) == 0 || throw(DimensionMismatch("The length of recvbuf and irecvbuf must be a multiple of the number of workers"))

    nactive = sum(pool.active)
    if nactive == 0
        return pool.repochs
    end

    # receive buffers for each worker
    rl = div(sizeof(irecvbuf), comm_size)
    irecvbufs = [view(reinterpret(UInt8, irecvbuf), (i-1)*rl+1:i*rl) for i in 1:comm_size]    
    recvbufs = [view(reinterpret(UInt8, recvbuf), (i-1)*rl+1:i*rl) for i in 1:comm_size]    

    # receive from all active workers
    MPI.Waitall!(pool.rreqs)
    for i in 1:length(pool.ranks)
        if pool.active[i]
            recvbufs[i] .= irecvbufs[i]
            pool.repochs[i] = pool.sepochs[i]
            pool.active[i] = false
            MPI.Wait!(pool.sreqs[i])
        end
    end

    return pool.repochs
end

end