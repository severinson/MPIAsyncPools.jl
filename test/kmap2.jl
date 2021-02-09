using MPI
using MPIStragglers
using Test

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const nworkers = MPI.Comm_size(comm) - 1
isroot() = MPI.Comm_rank(comm) == root
const data_tag = 0
const control_tag = 1

function shutdown(pool::AsyncPool)
    for i in pool.ranks
        MPI.Isend(zeros(1), i, control_tag, comm)
    end
end

function root_main()
    pool = AsyncPool(nworkers)
    @test pool.ranks == collect(1:nworkers)

    sendbuf = Vector{Float64}(undef, 1)
    isendbuf = zeros(eltype(sendbuf), nworkers*length(sendbuf))
    recvbuf = Vector{Float64}(undef, 3*nworkers)
    recvbufs = [view(recvbuf, (i-1)*3+1:i*3) for i in 1:nworkers]
    irecvbuf = copy(recvbuf)
    nwait = 2
    print("test $rank: root starting\n")

    # test that we have at least nwait responses from the current epoch in each iteration
    for epoch in 1:100
        sendbuf[1] = epoch
        delay = @elapsed begin
            repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm; nwait, tag=data_tag)
        end
        from_this_epoch = 0
        for i in 1:nworkers
            wrank, t, wepoch = recvbufs[i]

            # no tests if we've never received from this worker
            if repochs[i] == 0
                continue
            end
            if repochs[i] == epoch
                from_this_epoch += 1
            end

            # test that workers echo what was sent to them
            @test wepoch == repochs[i]
        end

        @test from_this_epoch >= nwait
        print("test $rank: response from $from_this_epoch workers in $delay seconds\n")
    end

    # test that all workers have returned after calling waitall!
    for epoch in 1:100
        repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm; nwait=1, tag=data_tag)
        repochs = waitall!(pool, recvbuf, irecvbuf)
        @test all(pool.active .== false)
    end

    # test using a custom function to determine when asyncmap! should return
    # (in this case that the first worker has returned)
    f = (epoch, repochs) -> repochs[1] == epoch
    for epoch in 101:200
        delay = @elapsed begin
            repochs = asyncmap!(pool, sendbuf, recvbuf, isendbuf, irecvbuf, comm; nwait=f, tag=data_tag)
        end
        @test repochs[1] == pool.epoch
        @test isapprox(delay, pool.latency[1], atol=1e-3)
        print("test $rank: response from the first worker in $delay ($(pool.latency[1])) seconds\n")
    end
    shutdown(pool)
end

function worker_main()
    recvbuf = Vector{Float64}(undef, 1)
    sendbuf = Vector{Float64}(undef, 3) # [rank, t, epoch]
    sendbuf[1] = rank
    crreq = MPI.Irecv!(zeros(1), root, control_tag, comm)

    t = 0
    while true
        t += 1

        rreq = MPI.Irecv!(recvbuf, root, data_tag, comm)
        index, _ = MPI.Waitany!([crreq, rreq])
        if index == 1 # exit message on control channel
            break
        end

        epoch = recvbuf[1]
        sendbuf[2] = t
        sendbuf[3] = epoch
        sleep(max(rand()/10, 0.005))
    
        sreq = MPI.Isend(sendbuf, root, data_tag, comm)
    end
end

if isroot()
    root_main()
else
    worker_main()
end
print("test $rank: exiting\n")
MPI.Barrier(comm)
if isroot()
    print("All tests pass\n")
end