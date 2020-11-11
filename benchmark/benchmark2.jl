using MPI
using MPIutils

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const nworkers = MPI.Comm_size(comm) - 1
isroot() = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1
const nelems = 10 # number of 64-bit floats sent in each direction per iteration and worker

function shutdown(pool::WorkerPool)
    for i in pool.ranks
        MPI.Isend(zeros(1), i, control_tag, comm)
    end
end

function root_main(k::Integer; pool::WorkerPool, nsamples=10000)
    sendbuf = Vector{Float64}(undef, nelems)
    isendbuf = Vector{Float64}(undef, nworkers*length(sendbuf))
    recvbuf = Vector{Float64}(undef, nworkers*nelems)
    irecvbuf = copy(recvbuf)
    for i in 1:nsamples
        epoch = i
        kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, k, epoch, pool, comm; tag=data_tag)
    end
    return
end

function worker_main()
    recvbuf = Vector{Float64}(undef, nelems)
    sendbuf = Vector{Float64}(undef, nelems)
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)
    while true
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            break
        end
        sendbuf .= recvbuf
        sreq = MPI.Isend(sendbuf, root, data_tag, comm)
    end
end

using Profile
if isroot()    
    pool = WorkerPool(nworkers)
    k = nworkers
    @profile root_main(k; pool) # run once to trigger compilation (ignore this one)
    Profile.clear()
    @profile root_main(k; pool)
    Profile.print(C=false)
    print("\n")
    shutdown(pool)
else
    worker_main()
end
MPI.Barrier(comm)