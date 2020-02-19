module Fields

export
    Face, Cell,
    AbstractField, Field, CellField, XFaceField, YFaceField, ZFaceField,
    interior, interiorparent,
    xnode, ynode, znode, location,
    set!,
    VelocityFields, TracerFields, tracernames, PressureFields, Tendencies

include("field.jl")
include("set!.jl")
include("field_tuples.jl")
include("show_fields.jl")

end
