"""
    module NumpyDates

Provides datetimes and timedeltas compatible with Numpy.

See: [`DateTime64`](@ref), [`InlineDateTime64`](@ref), [`TimeDelta64`](@ref) and
[`InlineTimeDelta64`](@ref).

These can generally be converted to/from their respective types in the `Dates` stdlib.
"""
module NumpyDates

using Dates: Dates, value

export Unit,
    YEARS,
    MONTHS,
    WEEKS,
    DAYS,
    HOURS,
    MINUTES,
    SECONDS,
    MILLISECONDS,
    MICROSECONDS,
    NANOSECONDS,
    PICOSECONDS,
    FEMTOSECONDS,
    ATTOSECONDS,
    AbstractDateTime64,
    InlineDateTime64,
    DateTime64,
    AbstractTimeDelta64,
    InlineTimeDelta64,
    TimeDelta64

include("common.jl")
include("Unit.jl")
include("AbstractDateTime64.jl")
include("InlineDateTime64.jl")
include("DateTime64.jl")
include("AbstractTimeDelta64.jl")
include("InlineTimeDelta64.jl")
include("TimeDelta64.jl")

end
