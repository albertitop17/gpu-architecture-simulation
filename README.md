# GPU Architecture Simulation

A simulation of a **GPU (Graphics Processing Unit)** architecture developed in **Python**. This project emulates the execution flow of parallel kernels across multiple Streaming Multiprocessors (SMs) and cores.

## 🏗️ Architectural Overview
The simulation replicates the hardware-software interface of modern GPUs:
* **Streaming Multiprocessors (SM):** Implemented using `multiprocessing.Process` to simulate independent hardware units.
* **CUDA Cores:** Emulated via `threading.Thread` within each SM to handle concurrent execution of threads.
* **Memory Hierarchy:** * **Global Memory (VRAM):** Shared across all SMs through the `GPUMemory` class.
    * **Shared/Local Memory:** Fast, per-SM memory implemented in `SMMemory`.
* **Synchronization:** Uses `threading.Barrier` to coordinate SIMT (Single Instruction, Multiple Threads) execution and prevent race conditions.

## 🧬 Implemented Kernels
The simulator includes several parallel algorithms to demonstrate different memory access patterns:
1. **Arithmetic Kernels:** Vector Increment and Vector Sum.
2. **Generalized Blur:** A stencil-based operation using a configurable radius and shared memory optimization.
3. **Dot Product (Reduction):** Uses atomic-style locking (`multiprocessing.Value.get_lock()`) to aggregate partial sums into a global result.

## 🛠️ Execution
To run the simulation and test different kernels, execute the main entry point:
```bash
python src/gpu.py
```
> **Note:** You can switch between kernels (INCR, SUMAR, DIFUMINAR, ESCALAR) by modifying the `KERNEL_A_PROBAR` variable in `gpu.py`.

## ✒️ Authorship
Developed by **Alberto Peña** and **Fabio Torres** (March 2026).
