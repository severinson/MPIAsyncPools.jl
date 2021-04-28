using MPI
using MPIAsyncPools
using Test

function parse_line(line::AbstractString; prefix="test")
    if !startswith(line, prefix)
        return -1, ""
    end
    line = line[length(prefix)+1:end]
    segments = split(line, ":")
    length(segments) == 2 || error("Unexpected colon in $line")
    rank = parse(Int, segments[1])
    output = strip(segments[2])
    rank, output
end

mpitestexec(n::Integer, file::String) = mpiexec(cmd -> read(`$cmd -n $n --mca btl ^openib julia --project $file`, String))

@testset "MPIAsyncPools.jl" begin
    n = 3
    for line in split(mpitestexec(n, "kmap1.jl"), "\n")
        rank, output = parse_line(line)
        if rank == -1
            continue
        end
        println((rank, line))
    end

    n = 3
    for line in split(mpitestexec(n, "kmap2.jl"), "\n")
        rank, output = parse_line(line)
        if rank == -1
            continue
        end
        println((rank, line))
    end    

    n = 10
    for line in split(mpitestexec(n, "kmap2.jl"), "\n")
        rank, output = parse_line(line)
        if rank == -1
            continue
        end
        println((rank, line))
    end

    println("No error messages indicates that all tests passed")
end