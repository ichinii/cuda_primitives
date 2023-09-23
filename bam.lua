src_dir = "src"
obj_dir = "obj"
bin_dir = "bin"

AddTool(function(s)
    s.cc.flags:Add("-Wall")
    s.cc.flags:Add("-Wextra")
    s.cc.flags_cxx:Add("--std=c++17")
    s.cc.includes:Add(src_dir)
    s.cu = {}
    s.cu.flags_compile = {}
    s.cu.flags_link = {}
    s.cu.flags = {}
    s.name = "default"

    local function objdir(s) return PathJoin(obj_dir, s.name) end

    s.cc.Output = function(s, input)
        input = input:gsub("^"..src_dir.."/", "")
        return PathJoin(objdir(s), PathBase(input))
    end

    s.compile.mappings.cu = function(s, input)
        local output = input:gsub("^"..src_dir.."/", "")
        output = PathJoin(objdir(s), PathBase(input))..".o"

        local flags = table.concat(TableFlatten({s.cu.flags_compile, s.cu.flags}), " ")
        AddJob(
            output,
            "nvcc "..input,
            "nvcc "..flags.." -c -o "..output.." "..input
        )
        AddDependency(output, input)
        --TODO: resolve include dependencies

        return output
    end

    s.link.exe = "/opt/cuda/bin/nvccxxxx"
    s.link.Driver = function(s, output, inputs)
        print(s)
        print(output)
        print(inputs)
    end
end)

function link(s, output, inputs)
    local flags = table.concat(TableFlatten({s.cu.flags_link, s.cu.flags}), " ")
    local input = table.concat(inputs, " ")
    AddJob(
        output,
        "Linking "..output,
        "nvcc "..flags.." -o "..output.." "..input
    )
    AddDependency(output, inputs)
    return output
end

function register(s)
    src_cpp = CollectRecursive(PathJoin(src_dir, "*.cpp"))
    src_cu = CollectRecursive(PathJoin(src_dir, "*.cu"))

    obj_cpp = Compile(s, src_cpp)
    obj_cu = Compile(s, src_cu)

    obj = TableFlatten({obj_cpp, obj_cu})
    bin = PathJoin(bin_dir, "b.out."..s.name)
    bin = link(s, bin, obj)

    local compile = "c_"..s.name
    PseudoTarget(compile, bin)

    local run = "r_"..s.name
    AddJob(
        run,
        "Running '"..bin.."'",
        "./"..bin
    )
    AddDependency(run, compile)

    PseudoTarget(s.name, run)
    return bin
end

-- target debug
s_debug = NewSettings()
s_debug.name = "debug"
s_debug.cc.flags:Add("-g")
table.insert(s_debug.cu.flags, "--debug")
table.insert(s_debug.cu.flags, "--device-debug")
register(s_debug)

-- target release
s_release = NewSettings()
s_release.name = "release"
s_release.cc.flags:Add("-O3")
table.insert(s_release.cu.flags, "--optimize 3")
bin_release = register(s_release)

-- target prof
AddJob(
    "prof",
    "Profiling '"..bin_release.."'",
    "nvprof "..bin_release
)
AddDependency("prof", bin_release)

DefaultTarget("debug")
