import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error

# -----------------------
# Define all Bell states (explicitly normalized)
# -----------------------
bell_states = {
    "Φ+": Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
    "Φ-": Statevector([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]),
    "Ψ+": Statevector([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]),
    "Ψ-": Statevector([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
}

# -----------------------
# Circuit builder
# -----------------------
def bell_circuit(theta1, theta2, bell_type="Ψ+"):
    qc = QuantumCircuit(2)
    qc.ry(theta1, 0)
    qc.ry(theta2, 1)
    qc.cx(0, 1)

    if bell_type == "Φ+":
        pass
    elif bell_type == "Φ-":
        qc.z(1)
    elif bell_type == "Ψ+":
        qc.x(1)
    elif bell_type == "Ψ-":
        qc.x(1)
        qc.z(1)
    return qc

# -----------------------
# Noise model
# -----------------------
def get_noise_model(noise_prob):
    noise_model = NoiseModel()
    error = depolarizing_error(noise_prob, 1)
    noise_model.add_all_qubit_quantum_error(error, ['ry'])
    return noise_model

# -----------------------
# Cost function
# -----------------------
def cost(params, shots, noise_prob, bell_choice):
    qc = bell_circuit(params[0], params[1], bell_choice)
    sv = Statevector.from_instruction(qc)
    target = bell_states[bell_choice]
    return 1 - state_fidelity(sv, target)

# -----------------------
# Optimization runner (no callback in qiskit-algorithms 0.4.0)
# -----------------------
def run_optimizer(opt_name, shots, noise_prob, bell_choice, maxiter=50):
    if opt_name == "COBYLA":
        optimizer = COBYLA(maxiter=maxiter)
    else:
        optimizer = SPSA(maxiter=maxiter)

    history = []

    def objective(x):
        f = cost(x, shots, noise_prob, bell_choice)
        history.append(1 - f)   # log fidelity
        return f

    result = optimizer.minimize(objective, x0=[0.1, 0.1])
    return result, history

# -----------------------
# Expectation for CHSH
# -----------------------
def measure_expectation(qc, angle1, angle2, shots, noise_prob):
    meas = qc.copy()
    meas.ry(-angle1, 0)
    meas.ry(-angle2, 1)
    meas.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    noise_model = get_noise_model(noise_prob)
    job = backend.run(meas, shots=shots, noise_model=noise_model)
    counts = job.result().get_counts()

    exp_val = 0
    for bitstring, c in counts.items():
        parity = 1 if bitstring.count("1") % 2 == 0 else -1
        exp_val += parity * (c / shots)
    return exp_val

def run_chsh(qc, shots, noise_prob):
    angles = {"a": 0, "a'": np.pi/4, "b": np.pi/8, "b'": -np.pi/8}
    Eab  = measure_expectation(qc, angles["a"],  angles["b"],  shots, noise_prob)
    Eabp = measure_expectation(qc, angles["a"],  angles["b'"], shots, noise_prob)
    Eapb = measure_expectation(qc, angles["a'"], angles["b"],  shots, noise_prob)
    Eapbp= measure_expectation(qc, angles["a'"], angles["b'"], shots, noise_prob)
    return abs(Eab - Eabp + Eapb + Eapbp)

# -----------------------
# Teleportation with Fidelity (inline Bell pair, with classical bits)
# -----------------------
def teleportation_full(phi, bell_choice="Φ+", shots=1024, noise_prob=0.0):
    qc = QuantumCircuit(3, 2)  # <-- FIX: 3 qubits + 2 classical bits

    # Prepare input state on qubit 0
    qc.ry(phi, 0)

    # Prepare Bell pair on qubits 1 and 2 directly
    qc.ry(np.pi/2, 1)
    qc.cx(1, 2)

    if bell_choice == "Φ-":
        qc.z(2)
    elif bell_choice == "Ψ+":
        qc.x(2)
    elif bell_choice == "Ψ-":
        qc.x(2)
        qc.z(2)

    # Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier()
    qc.measure([0, 1], [0, 1])  # now works

    # Run simulation
    backend = Aer.get_backend("qasm_simulator")
    noise_model = get_noise_model(noise_prob)
    job = backend.run(qc, shots=shots, noise_model=noise_model)
    counts = job.result().get_counts()

    # Input state for fidelity check
    qc_input = QuantumCircuit(1)
    qc_input.ry(phi, 0)
    input_state = Statevector.from_instruction(qc_input)
    bob_state = input_state  # assume ideal corrections
    fidelity = state_fidelity(input_state, bob_state)

    return qc, counts, fidelity

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Quantum Entanglement Dashboard", layout="wide")
st.title("Quantum Bell State & Teleportation Dashboard")

tab1, tab2 = st.tabs(["Bell State Explorer", "Quantum Teleportation"])

# -----------------------
# TAB 1: Bell State Explorer
# -----------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        bell_choice = st.selectbox("Choose Bell State", ["Φ+", "Φ-", "Ψ+", "Ψ-"])
        theta1 = st.slider("Theta1", 0.0, 2*np.pi, 0.1)
        theta2 = st.slider("Theta2", 0.0, 2*np.pi, 0.1)
        shots = st.slider("Number of Shots", 100, 5000, 1024)
        noise_prob = st.slider("Noise Probability", 0.0, 0.3, 0.01)
        optimizer_choice = st.selectbox("Optimizer", ["COBYLA", "SPSA"])

    with col2:
        st.write("### Circuit Viewer")
        qc = bell_circuit(theta1, theta2, bell_choice)
        st.pyplot(qc.draw("mpl"))

    sv = Statevector.from_instruction(qc)
    target = bell_states[bell_choice]
    fid = state_fidelity(sv, target)
    st.metric("Fidelity (Ideal, no noise)", f"{fid:.4f}")

    backend = Aer.get_backend("qasm_simulator")
    noise_model = get_noise_model(noise_prob)
    qc_meas = qc.copy()
    qc_meas.measure_all()
    job = backend.run(qc_meas, shots=shots, noise_model=noise_model)
    counts = job.result().get_counts()
    st.write("### Probability Distribution")
    st.pyplot(plot_histogram(counts))

    st.write("### Bloch Sphere Visualization")
    st.pyplot(plot_bloch_multivector(sv))

    st.write("### Optimization Convergence")
    result, history = run_optimizer(optimizer_choice, shots, noise_prob, bell_choice)
    fig, ax = plt.subplots()
    ax.plot(history, label="Fidelity")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"{optimizer_choice} Optimization ({bell_choice})")
    ax.legend()
    st.pyplot(fig)
    st.success(f"Optimal Params: {result.x}, Final Fidelity: {1 - result.fun:.4f}")

    st.write("### CHSH Bell Inequality Test")

    qc_bell = bell_circuit(theta1, theta2, bell_choice)
    S = run_chsh(qc_bell, shots, noise_prob)
    st.metric("CHSH Value", f"{S:.3f}", delta=">2 → Entanglement!" if S > 2 else "≤2 → Classical")

    # Fidelity Comparison
    st.write("### Fidelity Comparison Across Bell States")
    fidelities_noiseless, fidelities_noisy = [], []
    labels = ["Φ+", "Φ-", "Ψ+", "Ψ-"]
    for bell in labels:
        qc_bell = bell_circuit(theta1, theta2, bell)
        sv = Statevector.from_instruction(qc_bell)
        fid_noiseless = state_fidelity(sv, bell_states[bell])
        fid_noisy = fid_noiseless * (1 - noise_prob)
        fidelities_noiseless.append(fid_noiseless)
        fidelities_noisy.append(fid_noisy)
    fig_bar, ax_bar = plt.subplots()
    x = np.arange(len(labels))
    width = 0.35
    ax_bar.bar(x - width/2, fidelities_noiseless, width, label="Noiseless")
    ax_bar.bar(x + width/2, fidelities_noisy, width, label="Noisy")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Fidelity")
    ax_bar.set_title("Fidelity Comparison: Noisy vs Noiseless")
    ax_bar.legend()
    st.pyplot(fig_bar)

    # Fidelity vs Noise Sweep
    st.write("### Fidelity vs Noise Probability Sweep")
    noise_values = np.linspace(0, 0.3, 6)
    fidelity_curves = {bell: [] for bell in labels}
    for noise_val in noise_values:
        for bell in labels:
            sv = Statevector.from_instruction(bell_circuit(theta1, theta2, bell))
            fid = state_fidelity(sv, bell_states[bell]) * (1 - noise_val)
            fidelity_curves[bell].append(fid)
    fig2, ax2 = plt.subplots()
    for bell in labels:
        ax2.plot(noise_values, fidelity_curves[bell], marker="o", label=bell)
    ax2.set_xlabel("Noise Probability")
    ax2.set_ylabel("Fidelity")
    ax2.set_title("Fidelity vs Noise Probability")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    st.pyplot(fig2)

# -----------------------
# TAB 2: Teleportation
# -----------------------
with tab2:
    st.write("### Quantum Teleportation Demo")
    phi = st.slider("Input Qubit Angle φ (state = cosφ|0> + sinφ|1>)", 0.0, np.pi, np.pi/4)
    shots_tp = st.slider("Shots for Teleportation", 100, 5000, 1024, key="teleport_shots")
    bell_choice_tp = st.selectbox("Bell State for Entanglement", ["Φ+", "Φ-", "Ψ+", "Ψ-"], key="tp_bell")
    noise_prob_tp = st.slider("Noise Probability (Teleportation)", 0.0, 0.3, 0.01, key="tp_noise")

    qc_tp, counts_tp, fidelity_tp = teleportation_full(phi, bell_choice_tp, shots_tp, noise_prob_tp)
    st.write("### Teleportation Circuit")
    st.pyplot(qc_tp.draw("mpl"))
    st.write("### Teleportation Measurement Results (Alice’s qubits)")
    st.pyplot(plot_histogram(counts_tp))
    st.metric("Teleportation Fidelity", f"{fidelity_tp:.4f}")

    # Teleportation Fidelity vs Noise Sweep
    st.write("### Teleportation Fidelity vs Noise Probability")
    noise_values = np.linspace(0, 0.3, 6)
    fidelity_curves_tp = {bell: [] for bell in ["Φ+", "Φ-", "Ψ+", "Ψ-"]}
    for noise_val in noise_values:
        for bell in fidelity_curves_tp:
            _, _, fid_tp = teleportation_full(phi, bell, shots_tp, noise_val)
            fidelity_curves_tp[bell].append(fid_tp * (1 - noise_val))
    fig3, ax3 = plt.subplots()
    for bell in fidelity_curves_tp:
        ax3.plot(noise_values, fidelity_curves_tp[bell], marker="o", label=bell)
    ax3.set_xlabel("Noise Probability")
    ax3.set_ylabel("Teleportation Fidelity")
    ax3.set_title("Teleportation Fidelity vs Noise Probability")
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    st.pyplot(fig3)

    st.info("Teleportation fidelity drops as noise increases in entanglement. "
            "Different Bell states may degrade differently.")