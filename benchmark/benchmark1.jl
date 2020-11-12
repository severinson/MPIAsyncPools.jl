using MPI
using MPIutils
using Statistics

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const nworkers = MPI.Comm_size(comm) - 1
isroot() = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1
const nelems = 100 # number of 64-bit floats sent in each direction per iteration and worker

function shutdown(pool::StragglerPool)
    for i in pool.ranks
        MPI.Isend(zeros(1), i, control_tag, comm)
    end
end

function delay_samples(k::Integer; pool::StragglerPool, nsamples=100)
    sendbuf = Vector{Float64}(undef, nelems)
    isendbuf = Vector{Float64}(undef, nworkers*length(sendbuf))
    recvbuf = Vector{Float64}(undef, nworkers*nelems)
    irecvbuf = copy(recvbuf)
    samples = zeros(nsamples)
    for i in 1:nsamples
        epoch = i
        # kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, k, epoch, pool, comm; tag=data_tag)
        samples[i] = @elapsed kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, k, epoch, pool, comm; tag=data_tag)
    end
    samples
end

function root_main()
    pool = StragglerPool(nworkers)
    print("test $rank: root starting\n")    
    print("warming up\n")
    delay_samples(nworkers; pool)
    for k in 1:nworkers
        delay_samples(k; pool, nsamples=10)        
    end    

    print("starting test\n")
    samples = zeros(nworkers)
    for k in 1:nworkers
        k_samples = delay_samples(k; pool)
        t = mean(k_samples)
        t *= 1e6 # convert to μs
        t = round(t, digits=2)
        print("$k / $nworkers : $t μs, $(var(k_samples))\n")
        samples[k] = t
    end
    shutdown(pool)
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
        # sleep(max(rand()/10, 0.005))
        sreq = MPI.Isend(sendbuf, root, data_tag, comm)
    end
end

if isroot()
    root_main()
else
    worker_main()
end
MPI.Barrier(comm)