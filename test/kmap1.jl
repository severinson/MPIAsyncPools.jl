using MPI
using MPIStragglers
using Test

MPI.Init()
const comm = MPI.COMM_WORLD
const root = 0
const rank = MPI.Comm_rank(comm)
const nworkers = MPI.Comm_size(comm) - 1
isroot() = MPI.Comm_rank(comm) == root
const count = 1
const tag = 0

if isroot()
    pool = StragglerPool(nworkers)
    sendbuf = repeat([3.14], count)
    isendbuf = zeros(eltype(sendbuf), nworkers*length(sendbuf))
    recvbuf = Vector{Float64}(undef, nworkers)
    irecvbuf = copy(recvbuf)
    k = nworkers
    epoch = 0
    repochs = kmap!(sendbuf, recvbuf, isendbuf, irecvbuf, k, epoch, pool, comm; tag)
    @test recvbuf ≈ collect(1:nworkers)
else    
    recvbuf = Vector{Float64}(undef, 1)
    sendbuf = Vector{Float64}(undef, 1)
    sendbuf[1] = rank

    rreq = MPI.Irecv!(recvbuf, root, tag, comm)
    MPI.Wait!(rreq)
    @test recvbuf ≈ repeat([3.14], count)
    sreq = MPI.Isend(sendbuf, root, tag, comm)
    MPI.Wait!(sreq)
end

MPI.Barrier(comm)