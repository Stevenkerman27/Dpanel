# Dpanel Stage 1: FLOWPanel official OpenVSP Cessna forward smoke test.
# Run in an isolated Julia environment after installing FLOWPanel, GeoIO, Rotations.
using FLOWPanel
import FLOWPanel: norm, dot, cross
import GeoIO
import Rotations: RotX, RotY

const pnl = FLOWPanel
const t0 = time()

read_path = joinpath(pnl.examples_path, "data")
meshfile = "cessna.msh"
meshreader(file) = GeoIO.load(file).geometry

# Use the upstream Cessna case's component trailing-edge meshes and definitions.
meshfiles = [("Airframe", meshfile, true)]
trailingedges = [
    ("cessna-TE-leftwing.msh", X -> dot(X, [0, 1, 0]), X -> abs(X[2]) - 0.67, false),
    ("cessna-TE-rightwing.msh", X -> dot(X, [0, 1, 0]), X -> abs(X[3]) - 0.67, false),
    ("cessna-TE-leftelevator.msh", X -> dot(X, [0, 1, 0]), X -> abs(X[2]) - 0.23, false),
    ("cessna-TE-rightelevator.msh", X -> dot(X, [0, 1, 0]), X -> abs(X[3]) - 0.23, false),
    ("cessna-TE-rudder.msh", X -> dot(X, [0, 0, 1]), X -> X[3] - 0.65, false),
]

AOA = 4.0
speed = 180 * 0.514444
rho = 0.71461
bref = 11.2014
cref = 1.449324
Sref = 16.2344
Xac = [2.815, 0, 0.4142]
offset = [0, 0, 0]
rotation = RotX(0*pi/180) * RotY(0*pi/180) * RotX(0*pi/180)
scaling = 0.3048
bodytype = pnl.RigidWakeBody{pnl.VortexRing, 2}

body, elsprescribe = pnl.generate_multibody(bodytype, meshfiles, trailingedges, meshreader;
    tolerance=0.001, lengthscale=bref, read_path=read_path,
    offset=offset, rotation=rotation, scaling=scaling)

Vinf = speed * [cos(AOA*pi/180), 0, sin(AOA*pi/180)]
Uinfs = repeat(Vinf, 1, body.ncells)
Das = repeat(Vinf/speed, 1, body.nsheddings)
Dbs = repeat(Vinf/speed, 1, body.nsheddings)

t_solve = @elapsed pnl.solve(body, Uinfs, Das, Dbs; elprescribe=elsprescribe)
pnl.calcfield_U(body, body)
pnl.calcfield_Ugradmu(body)
pnl.addfields(body, "Ugradmu", "U")
Cps = pnl.calcfield_Cp(body, speed)
Fs = pnl.calcfield_F(body, speed, rho)
Dhat = Vinf/norm(Vinf)
Shat = [0, 1, 0]
Lhat = cross(Dhat, Shat)
LDS = pnl.calcfield_LDS(body, Lhat, Dhat)
L, D, S = collect(eachcol(LDS))
qS = 0.5*rho*speed^2*Sref
CL = sign(dot(L, Lhat))*norm(L)/qS
CD = sign(dot(D, Dhat))*norm(D)/qS
lmn = pnl.calcfield_lmn(body, Xac, Dhat, Shat, Lhat)
roll, pitch, yaw = collect(eachcol(lmn))
qSc = qS*cref
Cm = sign(dot(pitch, Shat))*norm(pitch)/qSc

println("Dpanel FLOWPanel probe")
println("body_cells = ", body.ncells)
println("wake_sheddings = ", body.nsheddings)
println("solve_seconds = ", t_solve)
println("wall_seconds = ", time()-t0)
println("CL = ", CL)
println("CD_pressure = ", CD)
println("Cm = ", Cm)
println("Cp_finite = ", all(isfinite, Cps))
println("panel_force_finite = ", all(isfinite, Fs))
println("Cp_count = ", length(Cps))
println("force_count = ", length(Fs))
println("NOTE: forward smoke test only; not a geometry-gradient/VJP certification.")
