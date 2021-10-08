# Iterative distributed computing example
# To run with 5 workers (rank 0 is the coordinator):
# mpirun -n 6 julia --project ./examples/iterative_example.jl

using MPI, MPIAsyncPools

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const isroot = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1
const COORDINATOR_TX_BYTES = 100
const WORKER_TX_BYTES = 100

function coordinator_main()
    
    # worker pool and communication buffers
    nworkers = MPI.Comm_size(comm) - 1
    nwait = 1
    pool = MPIAsyncPool(nworkers)
    
    # combined buffer for data received from all workers
    recvbuf = zeros(UInt8, nworkers*WORKER_TX_BYTES)

    # buffer for data being broadcast to the workers
    sendbuf = zeros(UInt8, COORDINATOR_TX_BYTES)

    # communication buffers for use internally in MPIAsyncPool
    isendbuf = similar(sendbuf, nworkers*length(sendbuf))
    irecvbuf = similar(recvbuf)

    # views into recvbuf corresponding to each worker
    n = div(length(recvbuf), nworkers)
    recvbufs = [view(recvbuf, (i-1)*n+1:i*n) for i in 1:nworkers]

    for epoch in 1:10
        sends = Vector{UInt8}("hello from coordinator on $(gethostname()), epoch $epoch")
        sendbuf[1:length(sends)] .= sends
        repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm; epoch, nwait, tag=data_tag)
        for i in 1:nworkers
            if repochs[i] == epoch
                recs = String(recvbufs[i])
                println("[coordinator]\t\treceived from worker $i:\t\t$recs")
            end
        end
    end

    # signal all workers to close
    waitall!(pool, recvbuf, irecvbuf)
    for i in pool.ranks
        MPI.Send(zeros(1), i, control_tag, comm)
    end
end

function worker_main()

    # control channel, to tell the workers when to exit
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)

    # buffer for data received from the coordinator
    recvbuf = zeros(UInt8, COORDINATOR_TX_BYTES)

    # buffer for data being sent to the coordinator
    sendbuf = zeros(UInt8, WORKER_TX_BYTES)

    # handle messages received from the coordinator until a message is received on the crreq channel
    i = 0
    while true
        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            MPI.Cancel!(rreq) # mark the rreq for cancellation
            MPI.Test!(rreq) # cleanup the data receive request
            break
        end
        sleep(rand()) # simulate performing a computation        
        recs = String(recvbuf[:])
        println("[worker $rank]\t\treceived from coordinator\t$recs")
        sends = Vector{UInt8}("hello from worker $rank on $(gethostname()), iteration $i")
        sendbuf[1:length(sends)] .= sends
        MPI.Send(sendbuf, root, data_tag, comm)
        i += 1
    end
end

if isroot
    coordinator_main()
else
    worker_main()
end
MPI.Barrier(comm)
MPI.Finalize()