@testitem "timedelta64" begin
    using Dates
    using CondaPkg
    CondaPkg.add("numpy")
    
    td = pyimport("numpy").timedelta64
    @testset for x in [
        -1_000_000_000,
        -1_000_000,
        -1_000,
        -1,
        0,
        1,
        1_000,
        1_000_000,
        1_000_000_000,
    ], (Unit, unit, pyunit) in [
        (Nanosecond, :nanoseconds, :ns),
        (Microsecond, :microseconds, :us),
        (Millisecond, :milliseconds, :ms),
        (Second, :seconds, :s),
        (Minute, :minutes, :m),
        (Hour, :hours, :h),
        (Day, :days, :D),
        (Week, :weeks, :W),
        (Month, :months, :M),
        (Year, :years, :Y),
    ]
        y = pytimedelta64(; [unit => x]...)
        y2 = pytimedelta64(Unit(x))
        @test pyeq(Bool, y, y2)
        @test pyeq(Bool, y, td(x, "$pyunit"))
    end
end

@testitem "datetime64" begin
    using Dates
    using CondaPkg
    CondaPkg.add("numpy")
    
    y = 2024
    m = 2
    d = 29
    h = 23
    min = 59
    s = 58
    ms = 999
    us = 998
    ns = 997

    date = DateTime(y, m, d, h, min, s, ms)
    pydate = pydatetime64(date)
    pydate2 = pydatetime64(year = y, month = m, day = d, hour = h, minute = min, second = s, millisecond = ms)
    dt = date - Second(0)
    pydate3 = pydatetime64(dt)
    @test pyeq(Bool, pydate, pydate2)
end