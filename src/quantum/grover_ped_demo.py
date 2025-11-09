# src/quantum/grover_ped_demo.py

import json
import math
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

from src.planning.config import PlanConfig
from src.planning.candidates import make_accel_profiles
from src.planning.evaluator import eval_candidate

SNAP_PATH = Path("snapshots/ped_scenario_tick.json")

# --- Bitstring ↔ Action mapping -------------------------------------------

BIT_TO_ACTION = {
    "00": "keep",
    "01": "comfort_brake",
    "10": "hard_brake",
    "11": "creep",
}
ACTION_TO_BIT = {v: k for k, v in BIT_TO_ACTION.items()}


# --- Lane builder (for snapshot replay) -----------------------------------

def fake_lane_from_snapshot(ego_loc, ego_vel, N_pts=400, ds=1.0):
    """Build a straight polyline in the direction of ego velocity."""
    v0 = math.hypot(ego_vel[0], ego_vel[1])
    heading = math.atan2(ego_vel[1], ego_vel[0]) if v0 > 0.1 else 0.0
    ux, uy = math.cos(heading), math.sin(heading)

    class P:
        __slots__ = ("x", "y", "z")

        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    pts = [
        P(
            ego_loc[0] + ux * i * ds,
            ego_loc[1] + uy * i * ds,
            ego_loc[2],
        )
        for i in range(N_pts)
    ]
    return pts, v0


# --- Quantum components ---------------------------------------------------

def grover_oracle_for_target(target_bits: str) -> QuantumCircuit:
    """
    Mark |target_bits> by phase inversion.

    Conventions:
      - target_bits is a string like '01', interpreted as |q1 q0>.
      - Qiskit qubit 0 is the LSB (rightmost bit), qubit 1 is MSB.
      - Statevector basis index i is formatted by format(i, "02b") -> 'q1q0'.
    """
    if len(target_bits) != 2:
        raise ValueError("target_bits must be a 2-bit string, e.g. '01'.")

    qc = QuantumCircuit(2)

    # Map bitstring 'b1b0' -> qubit1=b1, qubit0=b0
    bit_q1, bit_q0 = target_bits[0], target_bits[1]

    # X where target bit is 0 (to turn |target> into |11>)
    if bit_q0 == "0":
        qc.x(0)  # qubit 0 (LSB)
    if bit_q1 == "0":
        qc.x(1)  # qubit 1 (MSB)

    # Phase flip |11> via CZ
    qc.cz(0, 1)

    # Uncompute the Xs
    if bit_q0 == "0":
        qc.x(0)
    if bit_q1 == "0":
        qc.x(1)

    return qc


def grover_diffusion(n_qubits: int = 2) -> QuantumCircuit:
    """
    Grover diffusion operator for n_qubits (reflection about |s>).

    This is the standard construction:
        H^n X^n (multi-controlled Z) X^n H^n

    For n_qubits = 2, the multi-controlled Z reduces to a CZ between qubit 0 and 1.
    """
    qc = QuantumCircuit(n_qubits)

    # |s> -> |00..0>
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    # Multi-controlled Z about |11..1>
    # Implemented as H-Z-H on the last qubit, controlled by all others.
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    # |00..0> -> |s>
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    return qc

# --- Main driver ----------------------------------------------------------

def main():
    snap = json.loads(SNAP_PATH.read_text())

    cfg = PlanConfig(
        dt=snap["cfg"]["dt"],
        horizon_s=snap["cfg"]["horizon_s"],
        v_ref=snap["cfg"]["v_ref"],
        d_safe=snap["cfg"]["d_safe"],
    )

    ego_loc = snap["ego"]["loc"]
    ego_vel = snap["ego"]["vel"]
    ped_loc = snap["ped"]["loc"]
    ped_vel = snap["ped"]["vel"]
    ego_half_width = snap["ego"]["half_width"]
    t_world = snap["sim_time"]

    lane_points, v0 = fake_lane_from_snapshot(ego_loc, ego_vel)
    s0 = 0.0

    # Simple constant-velocity pedestrian predictor
    def ped_pred(t):
        dt = max(0.0, t - t_world)

        class P:
            __slots__ = ("x", "y", "z")

            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        return P(
            ped_loc[0] + ped_vel[0] * dt,
            ped_loc[1] + ped_vel[1] * dt,
            ped_loc[2],
        )

    # --- Classical evaluation --------------------------------------------
    all_profiles = make_accel_profiles(v0, cfg)
    profiles = {n: p for n, p in all_profiles.items() if n in ACTION_TO_BIT}

    print(f"\n[Offline] Testing {len(profiles)} profiles "
          f"at snapshot t={t_world:.2f}s")

    best_name, best_cost, best_diag = None, float("inf"), None

    for name, prof in profiles.items():
        valid, cost, diag = eval_candidate(
            prof, v0, s0, lane_points, ped_pred,
            cfg, t_world=t_world, ego_half_width=ego_half_width,
        )
        print(
            f"  - {name:14s} valid={valid} cost={cost:.2f} "
            f"dmin={diag.get('clearance_min', 0):.2f}"
        )
        if valid and cost < best_cost:
            best_name, best_cost, best_diag = name, cost, diag

    print(
        f"\n[Offline] Best classical profile: {best_name} "
        f"cost={best_cost:.2f} "
        f"dmin={best_diag.get('clearance_min', 0):.2f}"
    )

    # --- Quantum Grover search demo --------------------------------------
    target_bits = ACTION_TO_BIT[best_name]
    print(f"\n[Grover] Target action '{best_name}' encoded as |{target_bits}>")

    qc = QuantumCircuit(2, 2)

    # 1) Initialize in uniform superposition
    qc.h([0, 1])

    # 2–3) Grover iterations
    oracle = grover_oracle_for_target(target_bits)
    diffusion = grover_diffusion(2)

    num_iters = 1  # Optimal for 4 items

    for _ in range(num_iters):
        qc.compose(oracle, [0, 1], inplace=True)
        qc.compose(diffusion, [0, 1], inplace=True)

    # 4) Statevector diagnostics (no measurement)
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    print(f"\n[Grover] Statevector amplitudes after {num_iters} iterations:")
    for i, amp in enumerate(sv.data):
        bits = format(i, "02b")  # |q1 q0>
        print(f"  |{bits}>  amp={amp.real:+.3f}{amp.imag:+.3f}j")

    # 5) Measurement simulation
    qc.measure([0, 1], [0, 1])
    backend = Aer.get_backend("aer_simulator")
    qc = transpile(qc, backend)
    result = backend.run(qc, shots=512).result()
    counts = result.get_counts()

    print("\n[Grover] Measurement counts:")
    for bits, c in sorted(counts.items()):
        action = BIT_TO_ACTION.get(bits, "?")
        print(f"  |{bits}> ({action:14s}) -> {c} shots")

    best_bits = max(counts, key=counts.get)
    best_action = BIT_TO_ACTION[best_bits]
    print(
        f"\n[Grover] Most likely outcome: |{best_bits}> "
        f"→ action '{best_action}'"
    )


if __name__ == "__main__":
    main()